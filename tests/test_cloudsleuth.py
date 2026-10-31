"""
CloudSleuth Unit & Integration Tests.
Verifies data loading, statistical ranking, BARO adapter, and benchmark evaluation.
"""

import pytest
import os
import pandas as pd
import numpy as np

from cloudsleuth.data_loader import (
    list_available_cases,
    load_case_telemetry,
    get_ground_truth,
    CaseInfo,
    TelemetryBundle,
)
from cloudsleuth.engine import (
    analyze_telemetry_statistical,
    analyze_with_baro_baseline,
    AnalysisResult,
)
from cloudsleuth.evaluation import (
    evaluate_benchmark,
    BenchmarkEvaluationSummary,
)


def test_case_discovery():
    cases = list_available_cases()
    assert len(cases) > 0, "Should discover at least 1 real benchmark case"
    case_ids = [c.case_id for c in cases]
    assert "sample_baro_emailservice_mem" in case_ids, "Sample BARO case must be discovered"
    assert "multi_source_re2_ob" in case_ids, "RE2 multi-source case must be discovered"


def test_telemetry_loading_no_leakage():
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    assert isinstance(bundle, TelemetryBundle)
    assert not bundle.metrics_df.empty
    assert "time" in bundle.metrics_df.columns
    # Ensure ground truth is not leaked in the telemetry bundle
    assert not hasattr(bundle, "ground_truth")
    assert "ground_truth" not in bundle.metadata
    assert "fault" not in bundle.metadata


def test_ground_truth_isolation():
    gt_sample = get_ground_truth("sample_baro_emailservice_mem")
    assert gt_sample["service"] == "emailservice"
    assert gt_sample["fault"] == "mem"

    gt_ms = get_ground_truth("multi_source_re2_ob")
    assert gt_ms["service"] == "redis"
    assert gt_ms["fault"] == "disk"


def test_deterministic_statistical_engine():
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    
    # Run twice with same settings to guarantee determinism
    res1 = analyze_telemetry_statistical(bundle, zscore_threshold=3.0)
    res2 = analyze_telemetry_statistical(bundle, zscore_threshold=3.0)

    assert isinstance(res1, AnalysisResult)
    assert len(res1.ranked_services) > 0

    ranks1 = [c.service for c in res1.ranked_services]
    ranks2 = [c.service for c in res2.ranked_services]
    assert ranks1 == ranks2, "Analysis pipeline must be deterministic"

    # Verify top candidate diagnosis matches real emailservice_mem ground truth
    top_cand = res1.ranked_services[0]
    assert top_cand.service == "emailservice"
    assert "mem" in top_cand.primary_metric.lower()
    assert top_cand.confidence > 0.5
    assert len(top_cand.supporting_metrics) > 0


def test_multi_source_analysis():
    bundle = load_case_telemetry("multi_source_re2_ob")
    assert bundle.logts_df is not None and not bundle.logts_df.empty
    assert bundle.tracets_err_df is not None
    assert bundle.tracets_lat_df is not None

    res = analyze_telemetry_statistical(bundle)
    assert len(res.ranked_services) > 0
    top_cand = res.ranked_services[0]
    assert top_cand.service == "redis"
    assert res.uncertainty_level in ("Low", "Moderate")


def test_benchmark_evaluation_procedure():
    summary = evaluate_benchmark(case_limit=3, method="statistical")
    assert isinstance(summary, BenchmarkEvaluationSummary)
    assert summary.total_cases_evaluated == 3
    assert len(summary.evaluated_cases) == 3
    assert 0.0 <= summary.top_1_accuracy <= 1.0
    assert 0.0 <= summary.top_3_accuracy <= 1.0
    assert 0.0 <= summary.top_5_accuracy <= 1.0
    assert 0.0 <= summary.average_at_5 <= 1.0
    assert summary.avg_latency_ms > 0


# =============================================================
# GEMMA INTEGRATION TESTS
# =============================================================
import requests
from cloudsleuth.gemma_service import GemmaService, GemmaResponse
from unittest.mock import patch, MagicMock


def test_gemma_ground_truth_isolation():
    """Verify that Gemma prompt builders NEVER leak ground truth."""
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    result = analyze_telemetry_statistical(bundle)

    service = GemmaService()
    exp_prompt = service.build_explanation_prompt(result, bundle)
    rep_prompt = service.build_report_prompt(result, bundle)

    # Assert ground truth label helper is not called and terms aren't injected
    assert "ground_truth" not in exp_prompt.lower()
    assert "ground_truth" not in rep_prompt.lower()
    assert "actual root cause" not in exp_prompt.lower()
    assert "actual root cause" not in rep_prompt.lower()


def test_gemma_explanation_structure_and_fallback():
    """Test AI-powered incident explanation in offline fallback mode."""
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    result = analyze_telemetry_statistical(bundle)

    service = GemmaService(api_key="")
    response = service.explain_incident(result, bundle)

    assert isinstance(response, GemmaResponse)
    assert response.status == "fallback_deterministic"
    assert response.is_fallback is True
    assert "emailservice" in response.text.lower()

    # Verify all 6 mandatory sections are present
    assert "What Happened?" in response.text
    assert "Which Service Is Most Likely Responsible?" in response.text
    assert "Why Was This Service Selected?" in response.text
    assert "What Evidence Supports the Diagnosis?" in response.text
    assert "Which Other Services May Be Affected?" in response.text
    assert "What Should the Engineer Investigate Next?" in response.text


def test_gemma_incident_report_generation():
    """Test automatic incident report generation with 8 required sections."""
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    result = analyze_telemetry_statistical(bundle)

    service = GemmaService(api_key="")
    response = service.generate_incident_report(result, bundle)

    assert isinstance(response, GemmaResponse)
    assert response.status == "fallback_deterministic"
    assert "emailservice" in response.text.lower()

    # Verify all 8 mandatory sections exist
    required_sections = [
        "1. Incident Summary",
        "2. Incident Details",
        "3. Suspected Root Cause",
        "4. Supporting Telemetry Evidence",
        "5. Affected Services",
        "6. Incident Timeline",
        "7. Recommended Troubleshooting Steps",
        "8. Limitations and Uncertainties",
    ]
    for sec in required_sections:
        assert sec in response.text, f"Missing required report section: {sec}"


def test_gemma_api_failure_graceful_handling():
    """Test that API exceptions (e.g. network failure / 500) trigger graceful fallback without crashing."""
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    result = analyze_telemetry_statistical(bundle)

    # Provide a fake API key and mock requests.post to simulate a network outage
    service = GemmaService(api_key="fake-test-key")
    with patch("requests.post", side_effect=requests.RequestException("Connection timed out")):
        response = service.explain_incident(result, bundle)
        assert isinstance(response, GemmaResponse)
        assert response.status == "fallback_deterministic"
        assert response.is_fallback is True
        assert "Connection timed out" in (response.error_details or "")
        assert "What Happened?" in response.text

        rep_response = service.generate_incident_report(result, bundle)
        assert isinstance(rep_response, GemmaResponse)
        assert rep_response.status == "fallback_deterministic"
        assert rep_response.is_fallback is True
        assert "1. Incident Summary" in rep_response.text


def test_rca_engine_predictions_unmutated_by_gemma():
    """Verify that calling Gemma leaves RCA engine predictions and candidate ordering 100% untouched."""
    bundle = load_case_telemetry("sample_baro_emailservice_mem")
    result = analyze_telemetry_statistical(bundle)

    top_pred_before = result.ranked_services[0].service
    score_before = result.ranked_services[0].root_cause_score
    ranks_before = [c.service for c in result.ranked_services]

    service = GemmaService(api_key="")
    _ = service.explain_incident(result, bundle)
    _ = service.generate_incident_report(result, bundle)

    assert result.ranked_services[0].service == top_pred_before
    assert result.ranked_services[0].root_cause_score == score_before
    assert [c.service for c in result.ranked_services] == ranks_before


def test_gemma_multi_source_evidence_handling():
    """Verify that multi-source signals (logs and traces) are incorporated when present."""
    bundle = load_case_telemetry("multi_source_re2_ob")
    result = analyze_telemetry_statistical(bundle)

    service = GemmaService(api_key="")
    response = service.explain_incident(result, bundle)

    assert "redis" in response.text.lower()
    report = service.generate_incident_report(result, bundle)
    assert "redis" in report.text.lower()


# =============================================================
# DOCKER & DEPLOYMENT CONFIGURATION TESTS
# =============================================================
def test_docker_security_and_configuration():
    """Verify Dockerfile and .dockerignore adhere to security and deployment rules."""
    assert os.path.exists("Dockerfile"), "Dockerfile must exist"
    assert os.path.exists(".dockerignore"), ".dockerignore must exist"
    assert os.path.exists("docker-compose.yml"), "docker-compose.yml must exist"

    # Verify .dockerignore explicitly blocks .env to prevent secret leakage
    with open(".dockerignore", "r", encoding="utf-8") as f:
        dockerignore_content = f.read()
    assert ".env" in dockerignore_content, ".dockerignore must include .env to prevent baking secrets"

    # Verify Dockerfile parameters
    with open("Dockerfile", "r", encoding="utf-8") as f:
        dockerfile_content = f.read()

    assert "python:" in dockerfile_content and "slim" in dockerfile_content, "Must use lightweight slim base image"
    assert "8501" in dockerfile_content, "Must expose port 8501"
    assert "0.0.0.0" in dockerfile_content, "Must bind to 0.0.0.0 for external container accessibility"
    assert "requirements.txt" in dockerfile_content, "Must install dependencies from requirements.txt"

    # Verify docker-compose.yml maps port 8501
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        compose_content = f.read()
    assert "8501:8501" in compose_content, "docker-compose.yml must forward port 8501:8501"
    assert "cloudsleuth" in compose_content, "docker-compose.yml must define cloudsleuth service"
