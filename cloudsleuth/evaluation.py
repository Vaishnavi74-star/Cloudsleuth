"""
Benchmark Evaluation Module for CloudSleuth.
Strictly separates prediction from ground-truth evaluation.
Calculates official RCAEval metrics: Top-1 (Hit@1), Top-3 (Hit@3), Top-5 (Hit@5),
and Average@5 across verified benchmark test cases.
Never fabricates results or claims untested cases.
"""

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Ensure local RCAEval repo is available for official Evaluator comparison
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RCAEval_repo"))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

from cloudsleuth.data_loader import list_available_cases, load_case_telemetry, get_ground_truth
from cloudsleuth.engine import analyze_telemetry_statistical, analyze_with_baro_baseline


@dataclass
class CaseEvaluationRecord:
    case_id: str
    display_name: str
    system: str
    suite: str
    ground_truth_service: str
    ground_truth_fault: str
    predicted_top1: str
    predicted_top3: List[str]
    predicted_top5: List[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    rank_of_true_service: Optional[int]
    confidence_score: float
    latency_ms: float


@dataclass
class BenchmarkEvaluationSummary:
    method_name: str
    total_cases_evaluated: int
    evaluated_cases: List[CaseEvaluationRecord]
    hit_at_1_count: int
    hit_at_3_count: int
    hit_at_5_count: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    average_at_5: float
    avg_latency_ms: float
    suite_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


def evaluate_benchmark(
    case_limit: Optional[int] = 10,
    suite_filter: Optional[str] = None,
    method: str = "statistical",  # "statistical" or "baro"
    zscore_threshold: float = 3.0,
) -> BenchmarkEvaluationSummary:
    """
    Runs the benchmark evaluation over real tested cases.
    - Ground truth is retrieved ONLY AFTER predictions have been generated.
    - Accuracies and metrics are strictly calculated from the evaluated cases.
    """
    all_cases = list_available_cases()
    if suite_filter:
        all_cases = [c for c in all_cases if suite_filter.lower() in c.suite.lower()]

    if case_limit is not None and case_limit > 0:
        # Prioritize diverse cases: include multi-source, sample, and distinct fault types from RE1
        sample_cases = [c for c in all_cases if "sample" in c.case_id or "multi_source" in c.case_id]
        re1_cases = [c for c in all_cases if c not in sample_cases]
        
        # Take a balanced slice of RE1 cases (different faults)
        selected_re1 = []
        seen_faults = set()
        for c in re1_cases:
            parts = c.case_id.split("_")
            fault = parts[3] if len(parts) >= 4 else "unknown"
            if fault not in seen_faults:
                seen_faults.add(fault)
                selected_re1.append(c)
        for c in re1_cases:
            if c not in selected_re1:
                selected_re1.append(c)

        cases_to_run = (sample_cases + selected_re1)[:case_limit]
    else:
        cases_to_run = all_cases

    records: List[CaseEvaluationRecord] = []
    total_time_ms = 0.0

    for case_info in cases_to_run:
        # 1. Load telemetry (strictly without ground truth)
        bundle = load_case_telemetry(case_info.case_id)

        # 2. Run RCA prediction
        t0 = time.perf_counter()
        if method == "baro":
            result = analyze_with_baro_baseline(bundle)
        else:
            result = analyze_telemetry_statistical(bundle, zscore_threshold=zscore_threshold)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        total_time_ms += elapsed_ms

        pred_services = [c.service for c in result.ranked_services]
        top1 = pred_services[0] if len(pred_services) > 0 else "none"
        top3 = pred_services[:3]
        top5 = pred_services[:5]
        conf = result.ranked_services[0].confidence if result.ranked_services else 0.0

        # 3. Ground Truth is queried ONLY NOW for evaluation scoring
        gt = get_ground_truth(case_info.case_id)
        true_svc = gt["service"]
        true_fault = gt["fault"]

        rank_true = None
        for i, s in enumerate(pred_services):
            if s == true_svc:
                rank_true = i + 1
                break

        hit1 = true_svc == top1
        hit3 = true_svc in top3
        hit5 = true_svc in top5

        records.append(
            CaseEvaluationRecord(
                case_id=case_info.case_id,
                display_name=case_info.display_name,
                system=case_info.system,
                suite=case_info.suite,
                ground_truth_service=true_svc,
                ground_truth_fault=true_fault,
                predicted_top1=top1,
                predicted_top3=top3,
                predicted_top5=top5,
                hit_at_1=hit1,
                hit_at_3=hit3,
                hit_at_5=hit5,
                rank_of_true_service=rank_true,
                confidence_score=conf,
                latency_ms=round(elapsed_ms, 2),
            )
        )

    n = len(records)
    if n == 0:
        return BenchmarkEvaluationSummary(
            method_name=method,
            total_cases_evaluated=0,
            evaluated_cases=[],
            hit_at_1_count=0,
            hit_at_3_count=0,
            hit_at_5_count=0,
            top_1_accuracy=0.0,
            top_3_accuracy=0.0,
            top_5_accuracy=0.0,
            average_at_5=0.0,
            avg_latency_ms=0.0,
        )

    h1 = sum(1 for r in records if r.hit_at_1)
    h3 = sum(1 for r in records if r.hit_at_3)
    h5 = sum(1 for r in records if r.hit_at_5)

    ac1 = h1 / n
    ac3 = h3 / n
    ac5 = h5 / n

    # Compute official RCAEval Avg@5 = mean(AC@1, AC@2, AC@3, AC@4, AC@5)
    ac_ks = []
    for k in range(1, 6):
        hits_k = sum(1 for r in records if r.rank_of_true_service is not None and r.rank_of_true_service <= k)
        ac_ks.append(hits_k / n)
    avg_5 = float(np.mean(ac_ks))

    # Breakdown by suite
    suites = sorted(list(set(r.suite for r in records)))
    suite_breakdown = {}
    for s in suites:
        s_recs = [r for r in records if r.suite == s]
        sn = len(s_recs)
        sh1 = sum(1 for r in s_recs if r.hit_at_1)
        sh3 = sum(1 for r in s_recs if r.hit_at_3)
        sh5 = sum(1 for r in s_recs if r.hit_at_5)
        suite_breakdown[s] = {
            "count": sn,
            "top_1": round(sh1 / sn, 3),
            "top_3": round(sh3 / sn, 3),
            "top_5": round(sh5 / sn, 3),
        }

    return BenchmarkEvaluationSummary(
        method_name="CloudSleuth Statistical Engine" if method == "statistical" else "Official RCAEval BARO Baseline",
        total_cases_evaluated=n,
        evaluated_cases=records,
        hit_at_1_count=h1,
        hit_at_3_count=h3,
        hit_at_5_count=h5,
        top_1_accuracy=round(ac1, 3),
        top_3_accuracy=round(ac3, 3),
        top_5_accuracy=round(ac5, 3),
        average_at_5=round(avg_5, 3),
        avg_latency_ms=round(total_time_ms / n, 2),
        suite_breakdown=suite_breakdown,
        notes=[
            f"Evaluated {n} real benchmark test cases directly from disk.",
            "Predictions were generated blindly without access to ground-truth labels.",
            "Calculated using standard RCAEval metric definitions (Top-1, Top-3, Top-5, and Avg@5).",
        ],
    )
