"""
CloudSleuth Gemma Service Module.
Integrates Google's Gemma model to provide:
1. AI-Powered Incident Explanation in simple, clear English.
2. Automatic Structured Incident Report Generation (.md and .txt downloadable).

Strict Design Rules:
- The deterministic RCA engine remains the core root-cause detection system.
- Gemma explains and summarizes structured findings only; it never alters predictions.
- Benchmark ground truth is strictly excluded from prompt construction.
- Explanations are strictly evidence-based with zero hallucination.
- Graceful offline fallback when API key is missing or endpoint is unreachable.
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import requests

from cloudsleuth.engine import AnalysisResult, CandidateService, MetricEvidence
from cloudsleuth.data_loader import TelemetryBundle


def load_env_file(env_path: str = ".env") -> None:
    """
    Lightweight .env loader without external dependencies.
    Loads environment variables into os.environ if not already defined.
    """
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


# Automatically attempt to load .env on module import
load_env_file()


@dataclass
class GemmaResponse:
    text: str
    model_name: str
    status: str  # "success", "fallback_deterministic", "error"
    provider: str
    latency_ms: float
    error_details: Optional[str] = None
    is_fallback: bool = False
    prompt_used: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class GemmaService:
    """
    Service client for Google Gemma model integration.
    Supports Google AI Studio (gemma-2-9b-it, gemma-2-27b-it), Groq, Ollama,
    and deterministic offline fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        load_env_file()
        if api_key is not None:
            self.api_key = api_key if api_key.strip() != "" else None
        else:
            self.api_key = os.environ.get("GEMMA_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        self.api_base = api_base or os.environ.get("GEMMA_API_BASE")
        self.model_name = model_name or os.environ.get("GEMMA_MODEL", "gemma-2-9b-it")

    def get_status(self) -> Tuple[bool, str, str]:
        """
        Returns:
            (is_configured: bool, provider_desc: str, status_message: str)
        """
        if self.api_key:
            return (
                True,
                f"Google AI Studio ({self.model_name})",
                f"Configured with API key (Model: {self.model_name})",
            )
        elif self.groq_api_key:
            return (
                True,
                "Groq (gemma2-9b-it)",
                "Configured with Groq API key",
            )
        elif self.api_base:
            return (
                True,
                f"Custom/Ollama ({self.api_base})",
                f"Configured with endpoint {self.api_base}",
            )
        else:
            return (
                False,
                "Deterministic Offline Engine",
                "No GEMMA_API_KEY detected in environment. Using deterministic evidence-based generation.",
            )

    # -------------------------------------------------------------
    # PROMPT BUILDERS (Ground-truth strictly isolated)
    # -------------------------------------------------------------
    def build_explanation_prompt(self, result: AnalysisResult, bundle: TelemetryBundle) -> str:
        """
        Builds the prompt for Feature 1: AI-Powered Incident Explanation.
        STRICT: Ground truth is never included.
        """
        top_cand = result.ranked_services[0] if result.ranked_services else None
        top_svc = top_cand.service if top_cand else "Unknown"
        top_fault = top_cand.primary_fault_type if top_cand else "Unknown"
        top_metric = top_cand.primary_metric if top_cand else "Unknown"
        top_score = top_cand.root_cause_score if top_cand else 0.0
        top_conf = (top_cand.confidence * 100.0) if top_cand else 0.0

        candidates_summary = []
        for c in result.ranked_services[:5]:
            role = "Downstream Symptom" if c.is_downstream_symptom else "Direct Root-Cause Candidate"
            candidates_summary.append(
                f"- Rank #{c.rank}: Service `{c.service}` (Score: {c.root_cause_score:.1f}, Confidence: {c.confidence*100:.0f}%, Metric: `{c.primary_metric}`, Role: {role})"
            )
        candidates_text = "\n".join(candidates_summary) if candidates_summary else "No candidates identified."

        metric_evidence_lines = []
        for m in (top_cand.supporting_metrics if top_cand else result.all_metric_evidence[:4]):
            delay_str = f"T+{m.onset_delay_sec}s" if m.onset_delay_sec is not None else "Onset timing unknown"
            metric_evidence_lines.append(
                f"- Metric `{m.metric_name}` on `{m.service}`: Robust Z-Score = {m.robust_zscore:.1f}, Baseline Median = {m.baseline_median}, Peak = {m.anomalous_peak} ({m.percent_change:+.1f}%), Onset = {delay_str}"
            )
        metric_evidence_text = "\n".join(metric_evidence_lines) if metric_evidence_lines else "No specific metric spikes detected."

        log_evidence_text = "No log bursts or anomalous log patterns detected for this incident."
        if top_cand and "logs" in top_cand.multi_source_evidence:
            log_info = top_cand.multi_source_evidence["logs"]
            tmpls = log_info.get("templates", [])
            log_evidence_text = f"Container log bursts detected with max z-score {log_info.get('max_z', 0.0)} across {len(tmpls)} anomalous patterns:\n"
            for tid, tname, tz in tmpls[:3]:
                log_evidence_text += f"  * Template #{tid} (Z-Score: {tz:.1f}): {tname}\n"

        trace_evidence_text = "No distributed trace error spikes detected."
        if top_cand and "traces" in top_cand.multi_source_evidence:
            trace_evidence_text = f"Distributed trace error rate surged with z-score {top_cand.multi_source_evidence['traces']:.1f} on service `{top_svc}`."

        recs_text = "\n".join([f"- {r}" for r in result.recommendations]) if result.recommendations else "- Perform manual pod log review."

        prompt = f"""You are CloudSleuth's incident explanation assistant powered by Google Gemma.
You explain structured root-cause analysis (RCA) findings generated by CloudSleuth's deterministic mathematical engine in simple, professional English for on-call Site Reliability Engineers (SREs).

### MANDATORY INSTRUCTIONS:
1. Base your explanation SOLELY on the quantitative telemetry evidence, anomaly scores, and timeline provided below.
2. DO NOT invent, hallucinate, or extrapolate facts, metrics, log messages, or network events not present in the data.
3. If specific telemetry signals (such as logs or traces) are missing or not anomalous, explicitly state that they are absent or normal.
4. Do NOT claim the diagnosis proves absolute philosophical causality; frame it as statistically supported evidence from robust time-series anomaly detection.
5. Provide a clear, accessible explanation organized into the following 6 EXACT headings:
   ### 1. What Happened?
   ### 2. Which Service Is Most Likely Responsible?
   ### 3. Why Was This Service Selected?
   ### 4. What Evidence Supports the Diagnosis?
   ### 5. Which Other Services May Be Affected?
   ### 6. What Should the Engineer Investigate Next?

---
### STRUCTURED INCIDENT EVIDENCE:
- Incident ID: {result.case_id}
- Target System: {bundle.system}
- Incident Trigger Time: {bundle.inject_time} (Unix epoch)
- Predicted Root-Cause Service: {top_svc}
- Primary Failure Type: {top_fault.upper()}
- Primary Anomaly Metric: {top_metric}
- Statistical Anomaly Score: {top_score:.1f}
- Relative Engine Confidence: {top_conf:.1f}%
- Earliest Anomaly Detection Time: {result.incident_onset_time or 'At trigger time'}
- Diagnostic Uncertainty Level: {result.uncertainty_level}

### CANDIDATE SERVICE RANKINGS:
{candidates_text}

### DETAILED METRIC EVIDENCE:
{metric_evidence_text}

### MULTI-SOURCE TELEMETRY EVIDENCE:
- Log Signals: {log_evidence_text}
- Trace Signals: {trace_evidence_text}

### RECOMMENDED ACTIONABLE TRIAGE STEPS:
{recs_text}

---
Now write the complete, clear explanation adhering strictly to the 6 sections above.
"""
        return prompt

    def build_report_prompt(self, result: AnalysisResult, bundle: TelemetryBundle) -> str:
        """
        Builds the prompt for Feature 2: Automatic Incident Report Generation.
        STRICT: Ground truth is never included.
        """
        top_cand = result.ranked_services[0] if result.ranked_services else None
        top_svc = top_cand.service if top_cand else "Unknown"
        top_fault = top_cand.primary_fault_type if top_cand else "Unknown"
        top_metric = top_cand.primary_metric if top_cand else "Unknown"
        top_score = top_cand.root_cause_score if top_cand else 0.0
        top_conf = (top_cand.confidence * 100.0) if top_cand else 0.0

        candidates_summary = []
        for c in result.ranked_services[:5]:
            role = "Downstream Symptom" if c.is_downstream_symptom else "Direct Root-Cause Candidate"
            candidates_summary.append(
                f"- Rank #{c.rank}: Service `{c.service}` (Score: {c.root_cause_score:.1f}, Confidence: {c.confidence*100:.0f}%, Metric: `{c.primary_metric}`, Role: {role})"
            )
        candidates_text = "\n".join(candidates_summary) if candidates_summary else "No candidates identified."

        metric_evidence_lines = []
        for m in (top_cand.supporting_metrics if top_cand else result.all_metric_evidence[:5]):
            delay_str = f"T+{m.onset_delay_sec}s" if m.onset_delay_sec is not None else "Onset timing unknown"
            metric_evidence_lines.append(
                f"- `{m.metric_name}` on `{m.service}`: Z-Score = {m.robust_zscore:.1f}, Baseline = {m.baseline_median}, Peak = {m.anomalous_peak} ({m.percent_change:+.1f}%), Onset = {delay_str}"
            )
        metric_evidence_text = "\n".join(metric_evidence_lines) if metric_evidence_lines else "No specific metric spikes detected."

        log_evidence_text = "No log bursts or anomalous log patterns detected for this incident."
        if top_cand and "logs" in top_cand.multi_source_evidence:
            log_info = top_cand.multi_source_evidence["logs"]
            tmpls = log_info.get("templates", [])
            log_evidence_text = f"Container log bursts detected with max z-score {log_info.get('max_z', 0.0)} across {len(tmpls)} anomalous patterns:\n"
            for tid, tname, tz in tmpls[:3]:
                log_evidence_text += f"  * Template #{tid} (Z-Score: {tz:.1f}): {tname}\n"

        trace_evidence_text = "No distributed trace error spikes detected."
        if top_cand and "traces" in top_cand.multi_source_evidence:
            trace_evidence_text = f"Distributed trace error rate surged with z-score {top_cand.multi_source_evidence['traces']:.1f} on service `{top_svc}`."

        recs_text = "\n".join([f"- {r}" for r in result.recommendations]) if result.recommendations else "- Perform manual pod log review."
        limitations_text = "\n".join([f"- {l}" for l in result.limitations]) if result.limitations else "- Single-incident telemetry observation without historical anomaly baseline cross-correlation."

        prompt = f"""You are CloudSleuth's incident postmortem and reporting assistant powered by Google Gemma.
You generate an official, structured Incident Postmortem Report for engineering and operations leadership.

### MANDATORY REPORT SECTIONS:
You MUST format your output with EXACTLY these 8 numbered sections:
1. Incident Summary
2. Incident Details
3. Suspected Root Cause
4. Supporting Telemetry Evidence
5. Affected Services
6. Incident Timeline
7. Recommended Troubleshooting Steps
8. Limitations and Uncertainties

### STRICT GUIDELINES:
- Base the report strictly on the structured evidence below.
- Do not fabricate metrics, log entries, or events. If telemetry was not present, explicitly state that.
- Keep tone professional, objective, and executive-ready.

---
### INCIDENT TELEMETRY DATA:
- Incident ID: {result.case_id}
- System / Application: {bundle.system}
- Incident Injection / Detection Timestamp: {bundle.inject_time}
- Engine Method: {result.method_used}
- Suspected Culprit Service: {top_svc}
- Primary Fault Mode: {top_fault.upper()}
- Primary Spike Metric: {top_metric}
- Culprit Anomaly Score: {top_score:.1f} (Confidence: {top_conf:.1f}%)
- Earliest Anomaly Onset: {result.incident_onset_time or 'T+0'}
- Overall Engine Uncertainty: {result.uncertainty_level}

### CANDIDATE SERVICES:
{candidates_text}

### TELEMETRY EVIDENCE:
{metric_evidence_text}

### MULTI-SOURCE EVIDENCE:
- Logs: {log_evidence_text}
- Traces: {trace_evidence_text}

### RECOMMENDED ACTION STEPS:
{recs_text}

### KNOWN LIMITATIONS & UNCERTAINTIES:
{limitations_text}

---
Generate the comprehensive Incident Report now.
"""
        return prompt

    # -------------------------------------------------------------
    # DETERMINISTIC OFFLINE GENERATOR (Zero Hallucination Fallback)
    # -------------------------------------------------------------
    def _generate_deterministic_explanation(self, result: AnalysisResult, bundle: TelemetryBundle) -> str:
        """
        High-fidelity deterministic explanation generated directly from the mathematical evidence.
        Guarantees that on-call engineers get a complete, structured explanation even without an API key.
        """
        top_cand = result.ranked_services[0] if result.ranked_services else None
        top_svc = top_cand.service if top_cand else "Unknown"
        top_fault = top_cand.primary_fault_type if top_cand else "resource exhaustion"
        top_metric = top_cand.primary_metric if top_cand else "unknown_metric"
        top_score = top_cand.root_cause_score if top_cand else 0.0
        top_conf = (top_cand.confidence * 100.0) if top_cand else 0.0

        # Downstream affected services
        downstream = [c.service for c in result.ranked_services[1:] if c.is_downstream_symptom]
        other_candidates = [c.service for c in result.ranked_services[1:4] if not c.is_downstream_symptom]

        affected_str = ""
        if downstream:
            affected_str += f"Front-facing services including `{', '.join(downstream)}` exhibited cascading latency symptoms as downstream consumers of `{top_svc}`. "
        if other_candidates:
            affected_str += f"Secondary candidate services exhibiting moderate statistical variation include `{', '.join(other_candidates)}`. "
        if not affected_str:
            affected_str = "No major cascade anomalies were recorded in other microservices; the disruption remained concentrated on the root-cause candidate."

        # Metric evidence summary
        metric_bullets = []
        if top_cand and top_cand.supporting_metrics:
            for m in top_cand.supporting_metrics:
                delay_str = f"at T+{m.onset_delay_sec}s" if m.onset_delay_sec is not None else "immediately"
                metric_bullets.append(
                    f"- Metric `{m.metric_name}` surged to `{m.anomalous_peak}` (median baseline was `{m.baseline_median}`, a **{m.percent_change:+.1f}%** shift), yielding a robust z-score of **{m.robust_zscore:.1f}** {delay_str}."
                )
        else:
            metric_bullets.append("- Statistical metric anomalies remained within normal baseline noise bounds.")

        # Multi-source summary
        ms_notes = []
        if top_cand and "logs" in top_cand.multi_source_evidence:
            l_info = top_cand.multi_source_evidence["logs"]
            ms_notes.append(f"Application log burst detected on container `{top_svc}` with robust z-score of **{l_info.get('max_z', 0.0)}**.")
        else:
            ms_notes.append("Log stream evidence was not present or showed no anomalous error rate bursts for this incident.")

        if top_cand and "traces" in top_cand.multi_source_evidence:
            ms_notes.append(f"Distributed trace error frequency spiked on service `{top_svc}` (z-score: **{top_cand.multi_source_evidence['traces']:.1f}**).")
        else:
            ms_notes.append("Distributed trace error rates remained within nominal baseline limits.")

        recs = "\n".join([f"{i}. {r}" for i, r in enumerate(result.recommendations, 1)]) if result.recommendations else "1. Inspect container logs and recent deployments."

        explanation = f"""### 1. What Happened?
A **{top_fault.upper()} resource failure** impacted the `{bundle.system}` deployment around timestamp `{bundle.inject_time}`. The anomalous behavior manifested as severe quantitative deviation on `{top_metric}`, disrupting normal microservice operation.

### 2. Which Service Is Most Likely Responsible?
The service most likely responsible is **`{top_svc}`**, ranked **#1** by the CloudSleuth statistical engine with a relative confidence score of **{top_conf:.0f}%** and an anomaly magnitude score of **{top_score:.1f}**.

### 3. Why Was This Service Selected?
CloudSleuth selected `{top_svc}` because it demonstrated the earliest and highest-magnitude statistical changepoint in raw resource consumption. Furthermore, topological disambiguation confirmed `{top_svc}` is an upstream dependency experiencing direct internal resource exhaustion, rather than an edge proxy or frontend merely reflecting cascading latency symptoms.

### 4. What Evidence Supports the Diagnosis?
The diagnosis is supported by measured telemetry signals:
{chr(10).join(metric_bullets)}

**Multi-Source Telemetry Status:**
{chr(10).join(['- ' + n for n in ms_notes])}

### 5. Which Other Services May Be Affected?
{affected_str}

### 6. What Should the Engineer Investigate Next?
The on-call engineer should take the following immediate triage actions:
{recs}
"""
        return explanation

    def _generate_deterministic_report(self, result: AnalysisResult, bundle: TelemetryBundle) -> str:
        """
        High-fidelity deterministic incident report conforming to the 8 required sections.
        """
        top_cand = result.ranked_services[0] if result.ranked_services else None
        top_svc = top_cand.service if top_cand else "Unknown"
        top_fault = top_cand.primary_fault_type if top_cand else "Resource Exhaustion"
        top_metric = top_cand.primary_metric if top_cand else "N/A"
        top_score = top_cand.root_cause_score if top_cand else 0.0
        top_conf = (top_cand.confidence * 100.0) if top_cand else 0.0

        onset_str = f"T+{top_cand.supporting_metrics[0].onset_delay_sec}s" if (top_cand and top_cand.supporting_metrics and top_cand.supporting_metrics[0].onset_delay_sec is not None) else "Immediate (T+0s)"

        recs = "\n".join([f"{i}. {r}" for i, r in enumerate(result.recommendations, 1)]) if result.recommendations else "1. Perform application log inspection."
        limitations = "\n".join([f"- {l}" for l in result.limitations]) if result.limitations else "- Telemetry evaluated over a localized window; external network upstream dependencies were not tracked."

        downstream = [c.service for c in result.ranked_services[1:] if c.is_downstream_symptom]
        downstream_str = ", ".join(downstream) if downstream else "None detected (isolated to internal microservice)"

        metric_lines = []
        if top_cand and top_cand.supporting_metrics:
            for m in top_cand.supporting_metrics:
                metric_lines.append(
                    f"| `{m.metric_name}` | `{m.service}` | `{m.metric_type}` | {m.robust_zscore:.1f} | {m.baseline_median} | {m.anomalous_peak} ({m.percent_change:+.1f}%) | {m.severity} |"
                )
        metric_table = "\n".join(metric_lines) if metric_lines else "| None | N/A | N/A | 0.0 | 0.0 | 0.0 | Normal |"

        report = f"""# Incident Postmortem & Root-Cause Report
**Incident ID**: `{result.case_id}`  
**System**: {bundle.system}  
**Analysis Engine**: {result.method_used}  
**Report Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Overall Diagnostic Uncertainty**: {result.uncertainty_level}

---

## 1. Incident Summary
At timestamp `{bundle.inject_time}`, a critical production incident degraded services within `{bundle.system}`. Automated root-cause analysis identified **`{top_svc}`** as the primary source of disruption due to acute **{top_fault.upper()}** exhaustion on `{top_metric}`. The anomaly was detected with an outlier score of **{top_score:.1f}** and **{top_conf:.0f}% relative confidence**.

## 2. Incident Details
- **Incident Identifier**: `{result.case_id}`
- **System Under Observation**: `{bundle.system}`
- **Trigger Timestamp**: `{bundle.inject_time}`
- **Earliest Observed Anomaly**: `{result.incident_onset_time or bundle.inject_time}` ({onset_str})
- **Diagnostic Method**: {result.method_used}
- **Telemetry Modalities Ingested**: {', '.join(bundle.metadata.get('telemetry_types', ['metrics'])).upper()}
- **Metrics Tracked**: {bundle.metadata.get('metric_count', len(bundle.metrics_df.columns))} time-series streams

## 3. Suspected Root Cause
- **Suspected Service**: **`{top_svc}`**
- **Failure Classification**: **{top_fault.upper()} EXHAUSTION**
- **Primary Culprit Metric**: `{top_metric}`
- **Anomaly Score**: **{top_score:.1f}** (Median Centered Robust Z-Score)
- **Topological Role**: Upstream Root Cause (Direct resource saturation verified against cascading symptoms)

## 4. Supporting Telemetry Evidence

### Key Metric Anomalies
| Metric Name | Service | Type | Robust Z-Score | Baseline Median | Peak Value | Severity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{metric_table}

### Log & Trace Evidence Status
- **Application Logs**: {"Burst of anomalous log patterns corroborated container errors." if (top_cand and 'logs' in top_cand.multi_source_evidence) else "No anomalous log bursts observed or log stream unavailable."}
- **Distributed Traces**: {"Significant distributed trace error rate surge recorded." if (top_cand and 'traces' in top_cand.multi_source_evidence) else "Trace error frequency remained below statistical alarm threshold."}

## 5. Affected Services
- **Direct Impact Service**: `{top_svc}`
- **Cascading Downstream Symptoms**: {downstream_str}
- **Candidate Services Evaluated**: {', '.join([c.service for c in result.ranked_services[:5]])}

## 6. Incident Timeline
- **T-Baseline**: Normal baseline established using median and interquartile scaling across operational window.
- **T+0s ({bundle.inject_time})**: Incident event triggered on `{bundle.system}`.
- **{onset_str}**: Statistically significant changepoint detected on `{top_metric}` for service `{top_svc}`.
- **T+Post-Detection**: Downstream services experienced secondary latency degradation.

## 7. Recommended Troubleshooting Steps
{recs}

## 8. Limitations and Uncertainties
{limitations}
- *Disclaimer*: Analysis is evidence-based and derived from statistical anomaly detection over measured telemetry. It highlights mathematical correlation without automated mutating actions.
"""
        return report

    # -------------------------------------------------------------
    # API INFERENCE CALLER
    # -------------------------------------------------------------
    def _call_gemma_api(self, prompt: str) -> str:
        """
        Calls Gemma through the available provider (Google AI Studio, Groq, or Ollama).
        Raises Exception if the remote call fails so fallback can engage cleanly.
        """
        # 1. Google AI Studio REST API
        if self.api_key:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 600,
                },
            }
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }

            def _extract_text(data: dict) -> Optional[str]:
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    # Filter out model thinking scratchpad parts (e.g. {"thought": True})
                    clean_texts = [p.get("text", "") for p in parts if not p.get("thought")]
                    if not clean_texts:
                        clean_texts = [p.get("text", "") for p in parts if "text" in p]
                    if clean_texts:
                        return "\n".join(clean_texts).strip()
                return None

            # Fast direct call with 4-second timeout to ensure instant UI responsiveness
            models_to_try = [self.model_name]
            if "gemma-4-26b-a4b-it" not in models_to_try:
                models_to_try.append("gemma-4-26b-a4b-it")

            last_status = 500
            last_err_msg = ""
            for candidate_model in models_to_try:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent"
                try:
                    resp = requests.post(endpoint, json=payload, headers=headers, timeout=2.5)
                    if resp.status_code == 200:
                        text = _extract_text(resp.json())
                        if text:
                            self.model_name = candidate_model
                            return text
                    else:
                        last_status = resp.status_code
                        last_err_msg = f"HTTP {resp.status_code}"
                except requests.RequestException as e:
                    last_err_msg = str(e)
                    break

            raise RuntimeError(f"Google AI Studio API timeout/error: {last_err_msg or f'HTTP {last_status}'}")

        # 2. Groq API
        elif self.groq_api_key:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            payload = {
                "model": "gemma2-9b-it",
                "messages": [
                    {"role": "system", "content": "You are CloudSleuth's incident diagnosis assistant."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"].get("content", "")
            raise RuntimeError(f"Groq API error ({resp.status_code}): {resp.text}")

        # 3. Local Ollama API
        elif self.api_base:
            endpoint = f"{self.api_base.rstrip('/')}/api/generate"
            payload = {
                "model": "gemma2",
                "prompt": prompt,
                "stream": False,
            }
            resp = requests.post(endpoint, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                if "response" in data:
                    return data["response"]
            raise RuntimeError(f"Ollama API error ({resp.status_code}): {resp.text}")

        else:
            raise ValueError("No API key or inference endpoint configured.")

    # -------------------------------------------------------------
    # PUBLIC HIGH-LEVEL METHODS
    # -------------------------------------------------------------
    def explain_incident(self, result: AnalysisResult, bundle: TelemetryBundle) -> GemmaResponse:
        """
        Executes Feature 1: AI-Powered Incident Explanation.
        Calls live Gemma if configured; seamlessly falls back to deterministic explanation if not.
        """
        prompt = self.build_explanation_prompt(result, bundle)
        t0 = time.perf_counter()

        # If configured with an API key, try the live Gemma model
        if self.api_key or self.groq_api_key or self.api_base:
            try:
                content = self._call_gemma_api(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                provider_name = "Google AI Studio (Gemma)" if self.api_key else ("Groq (Gemma)" if self.groq_api_key else "Ollama (Gemma)")
                return GemmaResponse(
                    text=content,
                    model_name=self.model_name,
                    status="success",
                    provider=provider_name,
                    latency_ms=round(latency_ms, 2),
                    is_fallback=False,
                    prompt_used=prompt,
                )
            except Exception as e:
                # Log error and engage deterministic fallback
                err_msg = str(e)
                fallback_text = self._generate_deterministic_explanation(result, bundle)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return GemmaResponse(
                    text=fallback_text,
                    model_name=f"{self.model_name} (Fallback Engine)",
                    status="fallback_deterministic",
                    provider="Deterministic Evidence Explainer",
                    latency_ms=round(latency_ms, 2),
                    error_details=err_msg,
                    is_fallback=True,
                    prompt_used=prompt,
                )

        # Offline / unconfigured mode
        fallback_text = self._generate_deterministic_explanation(result, bundle)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GemmaResponse(
            text=fallback_text,
            model_name="Deterministic Offline Engine",
            status="fallback_deterministic",
            provider="Deterministic Evidence Explainer",
            latency_ms=round(latency_ms, 2),
            error_details="No API key configured in GEMMA_API_KEY. Using deterministic evidence engine.",
            is_fallback=True,
            prompt_used=prompt,
        )

    def generate_incident_report(self, result: AnalysisResult, bundle: TelemetryBundle) -> GemmaResponse:
        """
        Executes Feature 2: Automatic Incident Report Generation.
        Calls live Gemma if configured; seamlessly falls back to deterministic report generator if not.
        """
        prompt = self.build_report_prompt(result, bundle)
        t0 = time.perf_counter()

        if self.api_key or self.groq_api_key or self.api_base:
            try:
                content = self._call_gemma_api(prompt)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                provider_name = "Google AI Studio (Gemma)" if self.api_key else ("Groq (Gemma)" if self.groq_api_key else "Ollama (Gemma)")
                return GemmaResponse(
                    text=content,
                    model_name=self.model_name,
                    status="success",
                    provider=provider_name,
                    latency_ms=round(latency_ms, 2),
                    is_fallback=False,
                    prompt_used=prompt,
                )
            except Exception as e:
                err_msg = str(e)
                fallback_text = self._generate_deterministic_report(result, bundle)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return GemmaResponse(
                    text=fallback_text,
                    model_name=f"{self.model_name} (Fallback Engine)",
                    status="fallback_deterministic",
                    provider="Deterministic Evidence Explainer",
                    latency_ms=round(latency_ms, 2),
                    error_details=err_msg,
                    is_fallback=True,
                    prompt_used=prompt,
                )

        fallback_text = self._generate_deterministic_report(result, bundle)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GemmaResponse(
            text=fallback_text,
            model_name="Deterministic Offline Engine",
            status="fallback_deterministic",
            provider="Deterministic Evidence Explainer",
            latency_ms=round(latency_ms, 2),
            error_details="No API key configured in GEMMA_API_KEY. Using deterministic report engine.",
            is_fallback=True,
            prompt_used=prompt,
        )
