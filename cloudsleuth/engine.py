"""
CloudSleuth Deterministic & Explainable Root-Cause Analysis Engine.
Combines robust statistical anomaly detection, change-point onset timing,
microservice topological dependency heuristics, and multi-source telemetry fusion.
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from cloudsleuth.data_loader import TelemetryBundle

# Ensure RCAEval repo can be imported if needed for baseline comparison
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "RCAEval_repo"))
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)


@dataclass
class MetricEvidence:
    metric_name: str
    service: str
    metric_type: str
    robust_zscore: float
    baseline_median: float
    baseline_mad: float
    anomalous_peak: float
    percent_change: float
    onset_timestamp: Optional[int]
    onset_delay_sec: Optional[int]
    severity: str  # Critical, Warning, Normal


@dataclass
class CandidateService:
    rank: int
    service: str
    root_cause_score: float
    confidence: float
    primary_metric: str
    primary_fault_type: str
    onset_timestamp: Optional[int]
    evidence_summary: str
    supporting_metrics: List[MetricEvidence]
    is_downstream_symptom: bool = False
    multi_source_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    case_id: str
    method_used: str
    inject_time: int
    ranked_services: List[CandidateService]
    all_metric_evidence: List[MetricEvidence]
    incident_onset_time: Optional[int]
    earliest_detector_service: Optional[str]
    uncertainty_level: str  # Low, Moderate, High
    limitations: List[str]
    recommendations: List[str]
    analysis_settings: Dict[str, Any]


# Topology dependency map for Online Boutique microservices
# Frontend sits at the edge; backend services are called downstream.
TOPOLOGY_DEPTH = {
    "frontend": 0,
    "frontend-external": 0,
    "checkoutservice": 1,
    "cartservice": 2,
    "redis": 3,
    "paymentservice": 2,
    "emailservice": 2,
    "currencyservice": 2,
    "shippingservice": 2,
    "productcatalogservice": 2,
    "recommendationservice": 2,
    "adservice": 2,
}


def _classify_metric_type(metric_name: str) -> str:
    """Classify metric into CPU, MEM, DISK, LATENCY, WORKLOAD, LOSS/NETWORK, or OTHER."""
    name = metric_name.lower()
    if "cpu" in name:
        return "cpu"
    elif "mem" in name:
        return "mem"
    elif "disk" in name or "io" in name or "blk" in name:
        return "disk"
    elif "lat" in name:
        return "latency"
    elif "loss" in name or "drop" in name or "socket" in name:
        return "loss"
    elif "workload" in name or "req" in name or "qps" in name:
        return "workload"
    return "other"


def _compute_robust_scaler_score(normal_series: np.ndarray, anom_series: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes RobustScaler z-score matching RCAEval BARO formulation:
    scaler = RobustScaler().fit(baseline)
    zscores = scaler.transform(anomalous)[:, 0]
    score = max(zscores)
    Returns: (max_zscore, baseline_median, baseline_scale, peak_val)
    """
    a = normal_series.reshape(-1, 1)
    b = anom_series.reshape(-1, 1)
    scaler = RobustScaler().fit(a)
    zscores = scaler.transform(b)[:, 0]
    max_z = float(np.max(zscores)) if len(zscores) > 0 else 0.0
    
    median = float(scaler.center_[0]) if scaler.center_ is not None else float(np.median(normal_series))
    scale = float(scaler.scale_[0]) if scaler.scale_ is not None else float(np.std(normal_series) or 1.0)
    peak_val = float(np.max(anom_series)) if len(anom_series) > 0 else 0.0

    return max_z, median, scale, peak_val


def analyze_telemetry_statistical(
    bundle: TelemetryBundle,
    zscore_threshold: float = 3.0,
    window_minutes: Optional[int] = 10,
    prioritize_root_cause_topology: bool = True,
) -> AnalysisResult:
    """
    Deterministic, statistical RCA pipeline.
    1. Splits metrics into pre-incident baseline and post-incident anomaly windows.
    2. Uses RobustScaler (identical to RCAEval BARO) to score metric anomalies.
    3. Detects anomaly onset timestamps.
    4. Aggregates metric evidence into service-level candidate rankings.
    5. Applies topological root-cause vs cascading symptom disambiguation.
    6. Fuses log and trace anomalies if available.
    """
    df = bundle.metrics_df.copy()
    inject_time = bundle.inject_time

    # Split into normal and anomalous
    normal_df = df[df["time"] < inject_time]
    anomal_df = df[df["time"] >= inject_time]

    if normal_df.empty or anomal_df.empty:
        split_idx = len(df) // 2
        normal_df = df.iloc[:split_idx]
        anomal_df = df.iloc[split_idx:]
        inject_time = int(df["time"].iloc[split_idx])

    if window_minutes and window_minutes > 0:
        window_sec = window_minutes * 60
        normal_df = normal_df[normal_df["time"] >= (inject_time - window_sec)]
        anomal_df = anomal_df[anomal_df["time"] <= (inject_time + window_sec)]

    all_evidence: List[MetricEvidence] = []
    service_metric_map: Dict[str, List[MetricEvidence]] = {svc: [] for svc in bundle.service_names}

    metric_cols = [c for c in df.columns if c not in ("time", "time.1", "timestamp")]

    earliest_onset = None
    earliest_detector = None

    for col in metric_cols:
        parts = col.split("_")
        svc = parts[0].replace("-db", "") if len(parts) >= 2 else "system"
        mtype = _classify_metric_type(col)

        base_vals = normal_df[col].to_numpy()
        anom_vals = anomal_df[col].to_numpy()
        anom_times = anomal_df["time"].to_numpy()

        if len(base_vals) == 0 or len(anom_vals) == 0:
            continue

        # Use RobustScaler standard across RCAEval benchmark
        robust_z, median_base, scale_base, peak_val = _compute_robust_scaler_score(base_vals, anom_vals)

        # Percentage change
        pct_change = (
            ((peak_val - median_base) / (abs(median_base) if abs(median_base) > 1e-4 else 1.0)) * 100.0
        )

        # Onset timestamp detection (first timestep exceeding threshold)
        threshold_val = median_base + (zscore_threshold * scale_base)
        exceed_indices = np.where(anom_vals > threshold_val)[0]
        if len(exceed_indices) > 0:
            onset_ts = int(anom_times[exceed_indices[0]])
            onset_delay = onset_ts - inject_time
        else:
            onset_ts = None
            onset_delay = None

        if onset_ts is not None:
            if earliest_onset is None or onset_ts < earliest_onset:
                earliest_onset = onset_ts
                earliest_detector = svc

        severity = "Critical" if robust_z >= 10.0 else ("Warning" if robust_z >= zscore_threshold else "Normal")

        ev = MetricEvidence(
            metric_name=col,
            service=svc,
            metric_type=mtype,
            robust_zscore=round(float(robust_z), 2),
            baseline_median=round(float(median_base), 3),
            baseline_mad=round(float(scale_base), 3),
            anomalous_peak=round(float(peak_val), 3),
            percent_change=round(float(pct_change), 1),
            onset_timestamp=onset_ts,
            onset_delay_sec=onset_delay,
            severity=severity,
        )
        all_evidence.append(ev)
        if svc not in service_metric_map:
            service_metric_map[svc] = []
        service_metric_map[svc].append(ev)

    # Multi-source log evidence fusion
    log_evidence_map: Dict[str, Dict[str, Any]] = {}
    if bundle.logts_df is not None and not bundle.logts_df.empty:
        log_df = bundle.logts_df
        l_normal = log_df[log_df["time"] < inject_time]
        l_anomal = log_df[log_df["time"] >= inject_time]
        for col in log_df.columns:
            if col == "time":
                continue
            b_vals = l_normal[col].to_numpy()
            a_vals = l_anomal[col].to_numpy()
            if len(b_vals) > 0 and len(a_vals) > 0:
                l_z, l_med, l_scale, l_peak = _compute_robust_scaler_score(b_vals, a_vals)
                if l_z >= zscore_threshold:
                    svc_target = "unknown"
                    tmpl_text = f"Log Template #{col}"
                    if bundle.log_cluster_info and col in bundle.log_cluster_info:
                        info = bundle.log_cluster_info[col]
                        containers = info.get("container", [])
                        if containers:
                            svc_target = containers[0]
                        tmpl_text = info.get("template", tmpl_text)
                    
                    if svc_target not in log_evidence_map:
                        log_evidence_map[svc_target] = {"max_z": 0.0, "templates": []}
                    if l_z > log_evidence_map[svc_target]["max_z"]:
                        log_evidence_map[svc_target]["max_z"] = round(float(l_z), 2)
                    log_evidence_map[svc_target]["templates"].append((col, tmpl_text, round(float(l_z), 2)))

    # Trace error evidence fusion
    trace_evidence_map: Dict[str, float] = {}
    if bundle.tracets_err_df is not None and not bundle.tracets_err_df.empty:
        t_df = bundle.tracets_err_df
        t_norm = t_df[t_df["time"] < inject_time]
        t_anom = t_df[t_df["time"] >= inject_time]
        for col in t_df.columns:
            if col == "time":
                continue
            tb_vals = t_norm[col].to_numpy()
            ta_vals = t_anom[col].to_numpy()
            if len(tb_vals) > 0 and len(ta_vals) > 0:
                t_z, _, _, _ = _compute_robust_scaler_score(tb_vals, ta_vals)
                if t_z >= zscore_threshold:
                    svc_part = col.split("_")[0].replace("-db", "")
                    trace_evidence_map[svc_part] = max(trace_evidence_map.get(svc_part, 0.0), round(float(t_z), 2))

    # Aggregate to Service Candidates
    service_candidates: List[CandidateService] = []

    for svc, ev_list in service_metric_map.items():
        if not ev_list:
            continue

        # Sort service metrics by robust z-score
        sorted_ev = sorted(ev_list, key=lambda x: x.robust_zscore, reverse=True)
        top_ev = sorted_ev[0]

        # Base score from top metrics (mean of top 2)
        top_scores = [e.robust_zscore for e in sorted_ev[:2]]
        base_service_score = float(np.mean(top_scores)) if top_scores else 0.0

        # Multi-source bonuses
        log_bonus = 0.0
        if svc in log_evidence_map:
            log_bonus = min(log_evidence_map[svc]["max_z"] * 0.5, 20.0)

        trace_bonus = 0.0
        if svc in trace_evidence_map:
            trace_bonus = min(trace_evidence_map[svc] * 0.5, 20.0)

        # Topological Root Cause vs Cascading Symptom Disambiguation:
        # In microservices, frontend latency degradation is usually a downstream effect.
        # Direct resource exhaustion (CPU, Memory, Disk IO) in a backend service has higher causal priority.
        is_downstream_symptom = False
        topo_multiplier = 1.0
        if prioritize_root_cause_topology:
            depth = TOPOLOGY_DEPTH.get(svc, 2)
            if depth == 0 and top_ev.metric_type == "latency":
                is_downstream_symptom = True
                topo_multiplier = 0.55
            elif top_ev.metric_type in ("cpu", "mem", "disk"):
                topo_multiplier = 1.25

        total_score = (base_service_score + log_bonus + trace_bonus) * topo_multiplier

        # Earliest onset for this service
        svc_onsets = [e.onset_timestamp for e in sorted_ev if e.onset_timestamp is not None]
        svc_onset = min(svc_onsets) if svc_onsets else None

        # Human-readable evidence summary
        reason_parts = [
            f"Primary metric '{top_ev.metric_name}' spiked with z-score {top_ev.robust_zscore:.1f} ({top_ev.percent_change:+.1f}%)"
        ]
        if top_ev.onset_delay_sec is not None:
            reason_parts.append(f"onset at T+{top_ev.onset_delay_sec}s")
        if is_downstream_symptom:
            reason_parts.append("(flagged as downstream symptom of backend degradation)")
        if svc in log_evidence_map:
            reason_parts.append(f"Log error burst detected ({len(log_evidence_map[svc]['templates'])} anomalous patterns)")
        if svc in trace_evidence_map:
            reason_parts.append(f"Trace error spike detected (z-score: {trace_evidence_map[svc]:.1f})")

        evidence_str = "; ".join(reason_parts)

        ms_ev = {}
        if svc in log_evidence_map:
            ms_ev["logs"] = log_evidence_map[svc]
        if svc in trace_evidence_map:
            ms_ev["traces"] = trace_evidence_map[svc]

        service_candidates.append(
            CandidateService(
                rank=0,
                service=svc,
                root_cause_score=round(total_score, 2),
                confidence=0.0,
                primary_metric=top_ev.metric_name,
                primary_fault_type=top_ev.metric_type,
                onset_timestamp=svc_onset,
                evidence_summary=evidence_str,
                supporting_metrics=sorted_ev[:4],
                is_downstream_symptom=is_downstream_symptom,
                multi_source_evidence=ms_ev,
            )
        )

    # Sort candidates by root cause score descending
    service_candidates = sorted(service_candidates, key=lambda x: x.root_cause_score, reverse=True)

    # Compute relative confidence
    max_score = service_candidates[0].root_cause_score if service_candidates else 1.0
    second_score = service_candidates[1].root_cause_score if len(service_candidates) > 1 else 0.0

    for i, cand in enumerate(service_candidates):
        cand.rank = i + 1
        if max_score > 0:
            cand.confidence = round(float(cand.root_cause_score / (cand.root_cause_score + second_score + 1e-4)), 2)
        else:
            cand.confidence = 0.0

    # Uncertainty evaluation
    limitations: List[str] = []
    if not service_candidates or max_score < zscore_threshold:
        uncertainty_level = "High"
        limitations.append("No service exhibited telemetry anomalies exceeding the statistical threshold.")
    elif (max_score - second_score) < 2.0:
        uncertainty_level = "Moderate"
        limitations.append(
            f"Close competition between top candidates '{service_candidates[0].service}' and '{service_candidates[1].service}'."
        )
    else:
        uncertainty_level = "Low"

    # Actionable recommendations
    recommendations: List[str] = []
    if service_candidates:
        top = service_candidates[0]
        svc = top.service
        ft = top.primary_fault_type
        if ft == "mem":
            recommendations = [
                f"Inspect container memory usage & cgroup limits for `{svc}` (`kubectl describe pod -l app={svc}`).",
                f"Analyze Java/Go heap allocation profiles to trace potential memory leak or cache bloat.",
                f"Verify if Kubernetes OOMKilled restarts occurred near timestamp {bundle.inject_time}.",
                f"Temporarily increase memory limits or restart `{svc}` deployment as immediate triage.",
            ]
        elif ft == "cpu":
            recommendations = [
                f"Inspect CPU utilization & thread saturation on `{svc}` pods.",
                f"Check for sudden request traffic surges or expensive serialization loops.",
                f"Evaluate horizontal pod autoscaling (HPA) settings to scale out `{svc}` replicas.",
            ]
        elif ft == "disk":
            recommendations = [
                f"Inspect Disk I/O throughput and persistent volume IOPS limits on `{svc}` / `{svc}-db`.",
                f"Check database write amplification, transaction log retention, or slow sync queries.",
                f"Verify storage class volume read/write latency metrics.",
            ]
        elif ft == "latency" or ft == "delay":
            recommendations = [
                f"Inspect RPC call timeouts and circuit breaker trip events on `{svc}`.",
                f"Trace inter-service dependency graph to isolate blocking upstream network delay.",
                f"Review database connection pool utilization and keep-alive settings.",
            ]
        elif ft == "loss":
            recommendations = [
                f"Inspect Kubernetes network packet loss / iptables drops on node hosting `{svc}`.",
                f"Check service mesh / ingress proxy socket errors and MTU settings.",
            ]
        else:
            recommendations = [
                f"Investigate `{svc}` application logs around timestamp {bundle.inject_time}.",
                f"Review recent configuration deployments or environment variable updates.",
            ]

    return AnalysisResult(
        case_id=bundle.case_id,
        method_used="CloudSleuth Statistical RCA Engine (RobustScaler + Topology Disambiguation)",
        inject_time=bundle.inject_time,
        ranked_services=service_candidates,
        all_metric_evidence=sorted(all_evidence, key=lambda x: x.robust_zscore, reverse=True),
        incident_onset_time=earliest_onset,
        earliest_detector_service=earliest_detector,
        uncertainty_level=uncertainty_level,
        limitations=limitations,
        recommendations=recommendations,
        analysis_settings={
            "zscore_threshold": zscore_threshold,
            "window_minutes": window_minutes,
            "prioritize_root_cause_topology": prioritize_root_cause_topology,
        },
    )


def analyze_with_baro_baseline(bundle: TelemetryBundle) -> AnalysisResult:
    """
    Direct official RCAEval BARO execution for reproducible comparison.
    Uses official baro() or mmbaro() implementation from RCAEval.
    """
    from RCAEval.e2e.baro import baro, mmbaro
    from RCAEval.utility import read_data

    # Multi-source vs metric
    if bundle.logts_df is not None and not bundle.logts_df.empty:
        ms_data = {
            "metric": bundle.metrics_df,
            "logts": bundle.logts_df,
            "tracets_err": bundle.tracets_err_df if bundle.tracets_err_df is not None else pd.DataFrame(),
            "tracets_lat": bundle.tracets_lat_df if bundle.tracets_lat_df is not None else pd.DataFrame(),
        }
        raw_res = mmbaro(ms_data, inject_time=bundle.inject_time, dataset="re2-ob")
        method_name = "Official RCAEval MM-BARO (Multi-Source RobustScaler)"
    else:
        raw_res = baro(bundle.metrics_df, inject_time=bundle.inject_time, dataset="online-boutique")
        method_name = "Official RCAEval BARO Baseline (RobustScaler)"

    ranked_metrics = raw_res.get("ranks", [])

    # Group into service rankings
    seen_services = set()
    service_candidates: List[CandidateService] = []
    rank_counter = 1

    for metric in ranked_metrics:
        svc = metric.split("_")[0].replace("-db", "")
        if svc not in seen_services:
            seen_services.add(svc)
            service_candidates.append(
                CandidateService(
                    rank=rank_counter,
                    service=svc,
                    root_cause_score=round(max(10.0 - (rank_counter * 0.5), 1.0), 2),
                    confidence=round(max(1.0 - (rank_counter * 0.15), 0.1), 2),
                    primary_metric=metric,
                    primary_fault_type=_classify_metric_type(metric),
                    onset_timestamp=bundle.inject_time,
                    evidence_summary=f"Ranked #{rank_counter} by official RCAEval BARO baseline for metric '{metric}'.",
                    supporting_metrics=[],
                )
            )
            rank_counter += 1

    return AnalysisResult(
        case_id=bundle.case_id,
        method_used=method_name,
        inject_time=bundle.inject_time,
        ranked_services=service_candidates,
        all_metric_evidence=[],
        incident_onset_time=bundle.inject_time,
        earliest_detector_service=service_candidates[0].service if service_candidates else None,
        uncertainty_level="Low" if len(service_candidates) > 0 else "High",
        limitations=["Official BARO output is ranking-only; detailed metric-level attribution computed separately."],
        recommendations=[f"Triage top service candidate `{service_candidates[0].service}`."],
        analysis_settings={"method": "RCAEval BARO"},
    )
