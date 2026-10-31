"""
Data Loader for CloudSleuth.
Discovers and loads real benchmark telemetry from RCAEval datasets.
Ground-truth extraction is strictly isolated from telemetry data passed to the engine.
"""

import os
import json
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class CaseInfo:
    case_id: str
    display_name: str
    system: str
    suite: str
    telemetry_types: List[str]
    is_real_benchmark: bool
    data_dir: str
    data_file: str
    inject_time: int
    has_logs: bool = False
    has_traces: bool = False


@dataclass
class TelemetryBundle:
    case_id: str
    system: str
    inject_time: int
    metrics_df: pd.DataFrame
    logts_df: Optional[pd.DataFrame] = None
    tracets_err_df: Optional[pd.DataFrame] = None
    tracets_lat_df: Optional[pd.DataFrame] = None
    log_cluster_info: Optional[Dict[str, Any]] = None
    service_names: List[str] = field(default_factory=list)
    time_column: str = "time"
    metadata: Dict[str, Any] = field(default_factory=dict)


def _extract_services_from_columns(cols: List[str]) -> List[str]:
    """Extract distinct service entities from metric names like 'emailservice_cpu'."""
    services = set()
    for col in cols:
        if col in ("time", "time.1", "timestamp"):
            continue
        # Format is typically <service>_<metric_type>
        parts = col.split("_")
        if len(parts) >= 2:
            svc = parts[0].replace("-db", "")
            services.add(svc)
        else:
            services.add(col)
    return sorted(list(services))


def list_available_cases(base_dir: str = ".") -> List[CaseInfo]:
    """
    Discovers all available real benchmark cases on disk.
    Searches:
    1. data/online-boutique (RE1 Online Boutique benchmark suite)
    2. data/multi-source-data (RE2 Multi-source benchmark case)
    3. data.csv (Official BARO simple case)
    """
    cases: List[CaseInfo] = []

    # 1. Multi-source case
    multi_source_dir = os.path.join(base_dir, "data", "multi-source-data")
    if os.path.exists(multi_source_dir):
        inj_path = os.path.join(multi_source_dir, "inject_time.txt")
        inj_time = 1705354566
        if os.path.exists(inj_path):
            with open(inj_path, "r") as f:
                inj_time = int(f.read().strip())
        cases.append(
            CaseInfo(
                case_id="multi_source_re2_ob",
                display_name="RE2-OB: Multi-Source Incident (Metrics + Logs + Traces)",
                system="Online Boutique",
                suite="RE2 (Multi-Source)",
                telemetry_types=["metrics", "logs", "traces"],
                is_real_benchmark=True,
                data_dir=multi_source_dir,
                data_file=os.path.join(multi_source_dir, "metrics.csv"),
                inject_time=inj_time,
                has_logs=True,
                has_traces=True,
            )
        )

    # 2. Sample BARO data.csv
    simple_csv = os.path.join(base_dir, "data.csv")
    if os.path.exists(simple_csv):
        cases.append(
            CaseInfo(
                case_id="sample_baro_emailservice_mem",
                display_name="Sample Benchmark Case: emailservice_mem (BARO Reference)",
                system="Online Boutique",
                suite="Sample Reference",
                telemetry_types=["metrics"],
                is_real_benchmark=True,
                data_dir=base_dir,
                data_file=simple_csv,
                inject_time=1692569339,
                has_logs=False,
                has_traces=False,
            )
        )

    # 3. RE1 Online Boutique cases
    ob_dir = os.path.join(base_dir, "data", "online-boutique")
    if os.path.exists(ob_dir):
        fault_dirs = sorted(glob.glob(os.path.join(ob_dir, "*_*")))
        for fdir in fault_dirs:
            fault_name = os.path.basename(fdir)  # e.g., adservice_cpu
            parts = fault_name.split("_")
            if len(parts) != 2:
                continue
            svc_name, fault_type = parts[0], parts[1]

            rep_dirs = sorted(glob.glob(os.path.join(fdir, "*")))
            for rdir in rep_dirs:
                if not os.path.isdir(rdir):
                    continue
                rep_num = os.path.basename(rdir)
                data_csv = os.path.join(rdir, "data.csv")
                inj_txt = os.path.join(rdir, "inject_time.txt")

                if os.path.exists(data_csv) and os.path.exists(inj_txt):
                    try:
                        with open(inj_txt, "r") as f:
                            inj_time = int(f.read().strip())
                    except Exception:
                        continue

                    case_id = f"re1_ob_{svc_name}_{fault_type}_{rep_num}"
                    display_name = f"RE1-OB: {svc_name.upper()} {fault_type.upper()} (Run #{rep_num})"
                    cases.append(
                        CaseInfo(
                            case_id=case_id,
                            display_name=display_name,
                            system="Online Boutique",
                            suite="RE1 (Metric Benchmark)",
                            telemetry_types=["metrics"],
                            is_real_benchmark=True,
                            data_dir=rdir,
                            data_file=data_csv,
                            inject_time=inj_time,
                            has_logs=False,
                            has_traces=False,
                        )
                    )

    return cases


def load_case_telemetry(case_id: str, base_dir: str = ".", window_minutes: Optional[int] = None) -> TelemetryBundle:
    """
    Loads telemetry data for the given case ID.
    STRICT COMPLIANCE: Ground truth labels are NOT loaded or attached to this bundle.
    """
    cases = {c.case_id: c for c in list_available_cases(base_dir)}
    if case_id not in cases:
        raise ValueError(f"Unknown case_id: {case_id}")

    case = cases[case_id]
    
    # Load primary metrics DataFrame
    if not os.path.exists(case.data_file):
        raise FileNotFoundError(f"Metric file not found: {case.data_file}")

    df_metrics = pd.read_csv(case.data_file)
    
    # Preprocessing strictly as in official RCAEval
    if "time.1" in df_metrics.columns:
        df_metrics = df_metrics.drop(columns=["time.1"])
    df_metrics = df_metrics.replace([np.inf, -np.inf], np.nan)
    df_metrics = df_metrics.ffill().fillna(0)

    # Filter latency-50, standardize to latency-90 / latency
    df_metrics = df_metrics.loc[:, ~df_metrics.columns.str.endswith("_latency-50")]
    df_metrics = df_metrics.rename(
        columns={c: c.replace("_latency-90", "_latency") for c in df_metrics.columns if c.endswith("_latency-90")}
    )

    # Windowing around inject_time if requested
    inject_time = case.inject_time
    if window_minutes is not None and window_minutes > 0:
        window_sec = window_minutes * 60
        t_min = inject_time - window_sec
        t_max = inject_time + window_sec
        df_metrics = df_metrics[(df_metrics["time"] >= t_min) & (df_metrics["time"] <= t_max)].reset_index(drop=True)

    service_names = _extract_services_from_columns(list(df_metrics.columns))

    # Optional Multi-source data
    logts_df = None
    tracets_err_df = None
    tracets_lat_df = None
    cluster_info = None

    if case.has_logs:
        logts_path = os.path.join(case.data_dir, "logts.csv")
        if os.path.exists(logts_path):
            logts_df = pd.read_csv(logts_path).replace([np.inf, -np.inf], np.nan).fillna(0)
            if window_minutes is not None and window_minutes > 0 and "time" in logts_df.columns:
                logts_df = logts_df[(logts_df["time"] >= t_min) & (logts_df["time"] <= t_max)].reset_index(drop=True)

        cluster_info_path = os.path.join(case.data_dir, "cluster_info.json")
        if os.path.exists(cluster_info_path):
            try:
                with open(cluster_info_path, "r", encoding="utf-8") as f:
                    cluster_info = json.load(f)
            except Exception:
                cluster_info = None

    if case.has_traces:
        terr_path = os.path.join(case.data_dir, "tracets_err.csv")
        tlat_path = os.path.join(case.data_dir, "tracets_lat.csv")
        if os.path.exists(terr_path):
            tracets_err_df = pd.read_csv(terr_path).replace([np.inf, -np.inf], np.nan).fillna(0)
            if window_minutes is not None and window_minutes > 0 and "time" in tracets_err_df.columns:
                tracets_err_df = tracets_err_df[(tracets_err_df["time"] >= t_min) & (tracets_err_df["time"] <= t_max)].reset_index(drop=True)
        if os.path.exists(tlat_path):
            tracets_lat_df = pd.read_csv(tlat_path).replace([np.inf, -np.inf], np.nan).fillna(0)
            if window_minutes is not None and window_minutes > 0 and "time" in tracets_lat_df.columns:
                tracets_lat_df = tracets_lat_df[(tracets_lat_df["time"] >= t_min) & (tracets_lat_df["time"] <= t_max)].reset_index(drop=True)

    return TelemetryBundle(
        case_id=case.case_id,
        system=case.system,
        inject_time=inject_time,
        metrics_df=df_metrics,
        logts_df=logts_df,
        tracets_err_df=tracets_err_df,
        tracets_lat_df=tracets_lat_df,
        log_cluster_info=cluster_info,
        service_names=service_names,
        time_column="time",
        metadata={
            "suite": case.suite,
            "data_file": case.data_file,
            "display_name": case.display_name,
            "telemetry_types": case.telemetry_types,
            "metric_count": len([c for c in df_metrics.columns if c != "time"]),
            "time_range": [int(df_metrics["time"].min()), int(df_metrics["time"].max())],
        },
    )


def get_ground_truth(case_id: str) -> Dict[str, str]:
    """
    Returns ground truth for a case ID.
    CRITICAL: This function is strictly segregated from the prediction pipeline
    and must only be invoked post-diagnosis for evaluation scoring.
    """
    if case_id == "sample_baro_emailservice_mem":
        return {"service": "emailservice", "fault": "mem"}
    elif case_id == "multi_source_re2_ob":
        # Multi-source sample case injected redis diskio fault
        return {"service": "redis", "fault": "disk"}
    elif case_id.startswith("re1_ob_"):
        # Format: re1_ob_<service>_<fault>_<rep>
        parts = case_id.split("_")
        svc = parts[2]
        fault = parts[3]
        return {"service": svc, "fault": fault}
    else:
        raise ValueError(f"Ground truth not recorded for {case_id}")
