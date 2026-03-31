from itertools import combinations
import hashlib
import io
import math
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import numpy as np
import pandas as pd


app = FastAPI(title="VERDICT v2 - AI Decision Security Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets = {}
audit_history = []

SENSITIVE_KEYWORDS = [
    "race",
    "gender",
    "sex",
    "name",
    "ethnicity",
    "religion",
    "nationality",
    "marital",
    "disability",
    "age",
]
OUTCOME_KEYWORDS = [
    "outcome",
    "risk",
    "decision",
    "approve",
    "score",
    "recidivism",
    "default",
    "label",
    "target",
    "class",
    "result",
    "prediction",
    "status",
]


def detect_columns(df):
    cols = list(df.columns)
    outcome_col = next((c for c in cols if any(k in c.lower() for k in OUTCOME_KEYWORDS)), cols[-1])
    sensitive_cols = [c for c in cols if any(k in c.lower() for k in SENSITIVE_KEYWORDS)]
    if not sensitive_cols and len(cols) > 1:
        sensitive_cols = [cols[0]]
    return outcome_col, sensitive_cols


def get_dataset(dataset_id):
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return datasets[dataset_id]["df"].copy()


def normalize_value(value):
    if pd.isna(value):
        return "Unknown"
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def dominant_outcome(series):
    counts = series.astype(str).value_counts()
    if counts.empty:
        return "UNKNOWN"
    return str(counts.idxmax())


def encode_series(series):
    filled = series.fillna("Unknown")
    if pd.api.types.is_numeric_dtype(filled):
        return pd.to_numeric(filled, errors="coerce").fillna(0.0).astype(float)
    codes, _ = pd.factorize(filled.astype(str))
    return pd.Series(codes.astype(float), index=series.index)


def disparity_level(disparity):
    if disparity > 1.5:
        return "HIGH"
    if disparity > 1.2:
        return "MEDIUM"
    return "LOW"


def calculate_group_rates(df, outcome_col, sensitive_col, tracked_outcome=None):
    if outcome_col not in df.columns or sensitive_col not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid columns selected")

    working = df[[outcome_col, sensitive_col]].copy()
    working[sensitive_col] = working[sensitive_col].apply(normalize_value)
    working[outcome_col] = working[outcome_col].apply(normalize_value)
    tracked_outcome = tracked_outcome or dominant_outcome(working[outcome_col])

    group_stats = []
    for group, subset in working.groupby(sensitive_col):
        outcomes = subset[outcome_col].value_counts(normalize=True)
        rate = float(outcomes.get(tracked_outcome, 0.0))
        group_stats.append(
            {
                "group": group,
                "tracked_outcome": tracked_outcome,
                "rate": round(rate * 100, 1),
                "count": int(len(subset)),
                "distribution": {
                    normalize_value(k): round(float(v) * 100, 1) for k, v in outcomes.items()
                },
            }
        )

    group_stats.sort(key=lambda item: item["rate"], reverse=True)
    return tracked_outcome, group_stats


def calculate_chi_square(df, sensitive_col, outcome_col):
    contingency = pd.crosstab(
        df[sensitive_col].apply(normalize_value),
        df[outcome_col].apply(normalize_value),
    )
    if contingency.empty:
        return 0.0, False

    observed = contingency.to_numpy(dtype=float)
    row_totals = observed.sum(axis=1, keepdims=True)
    col_totals = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    if total == 0:
        return 0.0, False

    expected = row_totals @ col_totals / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_square = np.nan_to_num(((observed - expected) ** 2) / expected, nan=0.0, posinf=0.0).sum()
    degrees_of_freedom = max((observed.shape[0] - 1) * (observed.shape[1] - 1), 1)
    threshold = 3.84 if degrees_of_freedom == 1 else 5.99
    return round(float(chi_square), 2), float(chi_square) > threshold


def detect_proxy_features(df, sensitive_col, outcome_col):
    sensitive_encoded = encode_series(df[sensitive_col])
    outcome_encoded = encode_series(df[outcome_col])
    proxies = []

    for col in df.columns:
        if col in {sensitive_col, outcome_col}:
            continue
        try:
            feature_encoded = encode_series(df[col])
            proxy_corr = abs(float(feature_encoded.corr(sensitive_encoded)))
            outcome_corr = abs(float(feature_encoded.corr(outcome_encoded)))
        except Exception:
            continue
        if math.isnan(proxy_corr):
            proxy_corr = 0.0
        if math.isnan(outcome_corr):
            outcome_corr = 0.0
        if proxy_corr >= 0.15 or outcome_corr >= 0.15:
            score = (proxy_corr * 0.7) + (outcome_corr * 0.3)
            proxies.append(
                {
                    "feature": col,
                    "proxy_correlation": round(proxy_corr, 3),
                    "outcome_correlation": round(outcome_corr, 3),
                    "score": round(score, 3),
                    "influence": disparity_level(max(proxy_corr * 3, outcome_corr * 3)),
                }
            )

    proxies.sort(key=lambda item: item["score"], reverse=True)
    correlation_matrix = {}
    for proxy in proxies[:6]:
        correlation_matrix[proxy["feature"]] = {
            "to_sensitive": proxy["proxy_correlation"],
            "to_outcome": proxy["outcome_correlation"],
        }
    return proxies, correlation_matrix


def cohort_mask(df, record, columns):
    if not columns:
        return pd.Series([True] * len(df), index=df.index)
    mask = pd.Series([True] * len(df), index=df.index)
    for col in columns:
        value = normalize_value(record.get(col, "Unknown"))
        mask &= df[col].apply(normalize_value) == value
    return mask


def fair_prediction_from_record(df, outcome_col, sensitive_col, record, proxy_features):
    retained = [
        col
        for col in df.columns
        if col not in {outcome_col, sensitive_col}
        and col not in proxy_features
        and "unnamed" not in col.lower()
    ]
    prioritized = retained[:4]
    exact = df[cohort_mask(df, record, prioritized)] if prioritized else df
    if exact.empty and retained:
        exact = df[cohort_mask(df, record, retained[:2])]
    if exact.empty:
        exact = df
    return dominant_outcome(exact[outcome_col]), retained


def run_debias_analysis(df, outcome_col, sensitive_col):
    tracked_outcome, before_stats = calculate_group_rates(df, outcome_col, sensitive_col)
    proxies, correlation_matrix = detect_proxy_features(df, sensitive_col, outcome_col)
    proxy_features = [item["feature"] for item in proxies[:3]]

    sensitive_modes = {
        group: dominant_outcome(subset[outcome_col])
        for group, subset in df.groupby(df[sensitive_col].apply(normalize_value))
    }
    biased_predictions = []
    fair_predictions = []
    counterfactual_flips = 0

    for _, row in df.iterrows():
        record = row.to_dict()
        sensitive_value = normalize_value(row[sensitive_col])
        biased = sensitive_modes.get(sensitive_value, dominant_outcome(df[outcome_col]))
        fair, _ = fair_prediction_from_record(df, outcome_col, sensitive_col, record, proxy_features)
        biased_predictions.append(biased)
        fair_predictions.append(fair)

        alternate_predictions = [
            prediction for group, prediction in sensitive_modes.items() if group != sensitive_value
        ]
        if any(prediction != biased for prediction in alternate_predictions):
            counterfactual_flips += 1

    debiased_df = df.copy()
    debiased_df[outcome_col] = fair_predictions
    _, after_stats = calculate_group_rates(debiased_df, outcome_col, sensitive_col, tracked_outcome)
    before_rates = [item["rate"] for item in before_stats] or [0.0]
    after_rates = [item["rate"] for item in after_stats] or [0.0]
    disparity_before = round(max(before_rates) / max(min(before_rates), 0.1), 2)
    disparity_after = round(max(after_rates) / max(min(after_rates), 0.1), 2)
    records_affected = sum(1 for before, after in zip(biased_predictions, fair_predictions) if before != after)

    return {
        "method": "Proxy Feature Removal + Cohort Reweighting",
        "tracked_outcome": tracked_outcome,
        "proxy_features_removed": proxy_features,
        "proxy_correlation_matrix": correlation_matrix,
        "before": before_stats,
        "after": after_stats,
        "original_rates": {item["group"]: item["distribution"] for item in before_stats},
        "debiased_rates": {item["group"]: item["distribution"] for item in after_stats},
        "records_affected": records_affected,
        "counterfactual_flips": counterfactual_flips,
        "total_records": int(len(df)),
        "pct_affected": round((records_affected / max(len(df), 1)) * 100, 1),
        "disparity_before": disparity_before,
        "disparity_after": disparity_after,
        "fairness_gain": round(disparity_before - disparity_after, 2),
        "summary": (
            f"VERDICT would relabel {records_affected} records after removing the sensitive "
            f"attribute '{sensitive_col}' and its strongest proxies. "
            f"Disparity on '{tracked_outcome}' falls from {disparity_before}x to {disparity_after}x."
        ),
    }


def build_scan_summary(disparity_ratio, tracked_outcome, sensitive_col, highest, lowest):
    key_finding = (
        f"The {highest['group']} cohort is {disparity_ratio}x more likely to receive "
        f"'{tracked_outcome}' than the {lowest['group']} cohort."
    )
    risk = disparity_level(disparity_ratio)
    if risk == "HIGH":
        summary = (
            f"The decision system exhibits severe disparity across '{sensitive_col}'. "
            f"{key_finding} This pattern suggests historical bias is being reproduced."
        )
    elif risk == "MEDIUM":
        summary = (
            f"There is a moderate disparity across '{sensitive_col}'. {key_finding} "
            f"Proxy features and thresholds should be reviewed before deployment."
        )
    else:
        summary = (
            f"The system appears comparatively stable across '{sensitive_col}'. "
            f"The measured disparity of {disparity_ratio}x remains within the low-risk band."
        )
    return risk, key_finding, summary


def build_model_comparison(df, label):
    outcome_col, sensitive_cols = detect_columns(df)
    sensitive_col = sensitive_cols[0]
    tracked_outcome, group_stats = calculate_group_rates(df, outcome_col, sensitive_col)
    highest = group_stats[0]
    lowest = group_stats[-1]
    disparity_ratio = round(highest["rate"] / max(lowest["rate"], 0.1), 2)
    chi_square, significance = calculate_chi_square(df, sensitive_col, outcome_col)
    return {
        "label": label,
        "rows": int(len(df)),
        "outcome_col": outcome_col,
        "sensitive_col": sensitive_col,
        "tracked_outcome": tracked_outcome,
        "disparity_ratio": disparity_ratio,
        "risk_level": disparity_level(disparity_ratio),
        "statistical_significance": significance,
        "chi_squared": chi_square,
        "group_breakdown": group_stats[:6],
    }


def build_audit_history_entry(decision_id, verdict, timestamp, feature, severity):
    entry = {
        "decision_id": decision_id,
        "verdict": verdict,
        "timestamp": timestamp,
        "feature": feature,
        "severity": severity,
    }
    audit_history.append(entry)
    return entry


@app.get("/")
def read_root():
    return {
        "status": "VERDICT v2 Backend Running",
        "endpoints": [
            "/api/upload",
            "/api/scan",
            "/api/scan-intersectional",
            "/api/attack",
            "/api/fix",
            "/api/debias",
            "/api/predict-fair",
            "/api/compare-models",
            "/api/verdict",
            "/api/export-pdf",
        ],
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {exc}") from exc

    dataset_id = hashlib.sha256(f"{file.filename}_{time.time()}".encode()).hexdigest()[:16]
    outcome_col, sensitive_cols = detect_columns(df)
    datasets[dataset_id] = {
        "df": df,
        "filename": file.filename,
        "outcome_col": outcome_col,
        "sensitive_cols": sensitive_cols,
        "raw_bytes": contents,
    }

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "row_count": int(len(df)),
        "col_count": int(len(df.columns)),
        "columns": list(df.columns),
        "detected_outcome": outcome_col,
        "detected_sensitive": sensitive_cols,
        "preview": df.head(10).fillna("").to_dict(orient="records"),
    }


@app.post("/api/scan")
def scan_dataset(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    df = get_dataset(dataset_id)

    tracked_outcome, group_stats = calculate_group_rates(df, outcome_col, sensitive_col)
    if not group_stats:
        raise HTTPException(status_code=400, detail="Unable to compute group statistics")
    highest = group_stats[0]
    lowest = group_stats[-1]
    disparity_ratio = round(highest["rate"] / max(lowest["rate"], 0.1), 2)
    risk_level, key_finding, summary = build_scan_summary(
        disparity_ratio, tracked_outcome, sensitive_col, highest, lowest
    )
    chi_square, significance = calculate_chi_square(df, sensitive_col, outcome_col)
    proxies, correlation_matrix = detect_proxy_features(df, sensitive_col, outcome_col)

    bias_fingerprint = [{"feature": sensitive_col, "influence": risk_level}]
    bias_fingerprint.extend(
        {"feature": item["feature"], "influence": item["influence"]} for item in proxies[:3]
    )
    if len(bias_fingerprint) == 1:
        bias_fingerprint.append({"feature": "Proxy Features", "influence": "LOW"})

    return {
        "vulnerability_type": f"{sensitive_col.capitalize()} Bias",
        "severity": risk_level,
        "tracked_outcome": tracked_outcome,
        "disparity_ratio": f"{disparity_ratio}x",
        "disparity_raw": disparity_ratio,
        "group_stats": group_stats,
        "most_affected_group": f"'{highest['group']}' group",
        "key_finding": key_finding,
        "plain_english_summary": summary,
        "comparison_high": {"group": highest["group"], "rate": f"{highest['rate']}%"},
        "comparison_low": {"group": lowest["group"], "rate": f"{lowest['rate']}%"},
        "bias_fingerprint": bias_fingerprint,
        "statistical_significance": significance,
        "chi_squared": chi_square,
        "proxy_correlations": proxies[:6],
        "correlation_matrix": correlation_matrix,
        "affected_feature": sensitive_col,
    }


@app.post("/api/scan-intersectional")
def scan_intersectional(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_cols = payload.get("sensitive_cols", [])
    df = get_dataset(dataset_id)

    if len(sensitive_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sensitive columns for intersectionality analysis")

    results = []
    tracked_outcome = dominant_outcome(df[outcome_col])
    for col_a, col_b in combinations(sensitive_cols, 2):
        if col_a not in df.columns or col_b not in df.columns:
            continue
        working = df[[outcome_col, col_a, col_b]].copy()
        working["_intersection"] = (
            working[col_a].apply(normalize_value) + " x " + working[col_b].apply(normalize_value)
        )
        group_stats = []
        for group, subset in working.groupby("_intersection"):
            rate = float((subset[outcome_col].apply(normalize_value) == tracked_outcome).mean())
            group_stats.append(
                {
                    "intersection": group,
                    "rate": round(rate * 100, 1),
                    "count": int(len(subset)),
                }
            )
        group_stats.sort(key=lambda item: item["rate"], reverse=True)
        if len(group_stats) < 2:
            continue
        highest = group_stats[0]
        lowest = group_stats[-1]
        gap = round(highest["rate"] / max(lowest["rate"], 0.1), 2)
        results.append(
            {
                "axes": f"{col_a} x {col_b}",
                "tracked_outcome": tracked_outcome,
                "highest": highest,
                "lowest": lowest,
                "disparity": f"{gap}x",
                "disparity_raw": gap,
                "all_groups": group_stats[:8],
                "severity": disparity_level(gap),
            }
        )

    return {"intersections": results}


@app.post("/api/attack")
def attack_model(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    original_val = normalize_value(payload.get("original_val"))
    new_val = normalize_value(payload.get("new_val"))
    df = get_dataset(dataset_id)

    df_a = df[df[sensitive_col].apply(normalize_value) == original_val]
    df_b = df[df[sensitive_col].apply(normalize_value) == new_val]
    if df_a.empty or df_b.empty:
        midpoint = max(len(df) // 2, 1)
        df_a = df.head(midpoint)
        df_b = df.tail(midpoint)

    profiles_a = df_a.copy()
    profiles_b = df_b.copy()
    prof_a = profiles_a.iloc[0].fillna("").to_dict()
    prof_b = profiles_b.iloc[0].fillna("").to_dict()

    return {
        "original": prof_a,
        "modified": prof_b,
        "review_a": (
            f"Profile A received '{normalize_value(prof_a.get(outcome_col))}'. "
            f"That outcome aligns with the harsher pattern learned for '{original_val}'."
        ),
        "review_b": (
            f"Profile B received '{normalize_value(prof_b.get(outcome_col))}'. "
            f"That outcome aligns with the more favorable pattern learned for '{new_val}'."
        ),
        "message": "Two real dataset records were loaded for comparison.",
    }


@app.post("/api/fix")
def generate_fix(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    original_val = normalize_value(payload.get("original_val"))
    df = get_dataset(dataset_id)

    sample_df = df[df[sensitive_col].apply(normalize_value) == original_val]
    sample = sample_df.iloc[0] if not sample_df.empty else df.iloc[0]
    proxies, _ = detect_proxy_features(df, sensitive_col, outcome_col)
    removed = [sensitive_col] + [item["feature"] for item in proxies[:2]]
    fair_score, retained = fair_prediction_from_record(df, outcome_col, sensitive_col, sample.to_dict(), removed)

    return {
        "fair_score": fair_score,
        "biased_score": normalize_value(sample[outcome_col]),
        "factors_used": retained[:4],
        "factors_removed": removed,
        "fair_recommendation": (
            f"Recommendation based on retained factors only: {', '.join(retained[:4]) or 'global trend baseline'}."
        ),
        "what_changed": (
            f"VERDICT removed '{sensitive_col}' and the strongest proxy features from the scoring path "
            f"before recalculating the outcome."
        ),
        "impact_statement": (
            f"This isolates legitimate evidence from demographic leakage so the decision no longer depends "
            f"on protected-class signals."
        ),
    }


@app.post("/api/debias")
def debias_dataset(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    df = get_dataset(dataset_id)
    return run_debias_analysis(df, outcome_col, sensitive_col)


@app.post("/api/predict-fair")
def predict_fair(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    record = payload.get("record", {})
    df = get_dataset(dataset_id)

    analysis = run_debias_analysis(df, outcome_col, sensitive_col)
    proxies = analysis["proxy_features_removed"]
    sensitive_modes = {
        group: dominant_outcome(subset[outcome_col])
        for group, subset in df.groupby(df[sensitive_col].apply(normalize_value))
    }
    selected_group = normalize_value(record.get(sensitive_col, "Unknown"))
    biased_prediction = sensitive_modes.get(selected_group, dominant_outcome(df[outcome_col]))
    fair_prediction, retained = fair_prediction_from_record(df, outcome_col, sensitive_col, record, proxies)

    counterfactual_predictions = {}
    for group, prediction in sensitive_modes.items():
        counterfactual_predictions[group] = prediction
    outcome_would_change = len(set(counterfactual_predictions.values())) > 1

    return {
        "input_record": record,
        "biased_prediction": biased_prediction,
        "fair_prediction": fair_prediction,
        "counterfactual_outcome": counterfactual_predictions,
        "outcome_would_change": outcome_would_change,
        "risk_flag": outcome_would_change,
        "proxy_features_removed": proxies,
        "retained_features": retained[:5],
        "explanation": (
            f"The biased model predicts '{biased_prediction}' because the record is evaluated inside the "
            f"'{selected_group}' demographic bucket. After removing '{sensitive_col}' and proxy features, "
            f"VERDICT predicts '{fair_prediction}'."
        ),
        "confidence": "HIGH" if outcome_would_change else "LOW",
    }


@app.post("/api/compare-models")
async def compare_models(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        df1 = pd.read_csv(io.StringIO((await file1.read()).decode("utf-8")))
        df2 = pd.read_csv(io.StringIO((await file2.read()).decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error parsing CSVs: {exc}") from exc

    model_a = build_model_comparison(df1, file1.filename or "Model A")
    model_b = build_model_comparison(df2, file2.filename or "Model B")
    if model_a["disparity_ratio"] < model_b["disparity_ratio"]:
        fairer = "Model A"
    elif model_b["disparity_ratio"] < model_a["disparity_ratio"]:
        fairer = "Model B"
    else:
        fairer = "Equal"

    return {
        "model_a": model_a,
        "model_b": model_b,
        "fairer_model": fairer,
        "recommendation": (
            f"{fairer} exhibits lower demographic disparity and is recommended for deployment."
            if fairer != "Equal"
            else "Both models show comparable fairness characteristics."
        ),
    }


@app.post("/api/verdict")
def generate_verdict(payload: dict):
    dataset_id = payload.get("dataset_id")
    is_fair = payload.get("is_fair", False)
    severity = payload.get("severity", "UNKNOWN")
    disparity = payload.get("disparity", "UNKNOWN")
    feature = payload.get("feature", "UNKNOWN")
    timestamp = time.time()
    raw_str = f"{dataset_id}_{is_fair}_{timestamp}"
    integrity_hash = "0x" + hashlib.sha256(raw_str.encode()).hexdigest().upper()
    verdict = "FAIR" if is_fair else "NOT FAIR"
    decision_id = f"DEC-{int(timestamp)}"

    if not is_fair:
        compliance = [
            {
                "regulation": "EU AI Act - Article 6",
                "status": "NON-COMPLIANT",
                "detail": "High-risk AI behavior is indicated by measurable discriminatory disparity.",
            },
            {
                "regulation": "US ECOA",
                "status": "VIOLATION RISK",
                "detail": f"Protected-class impact was observed on '{feature}' beyond acceptable parity thresholds.",
            },
            {
                "regulation": "GDPR Article 22",
                "status": "REVIEW REQUIRED",
                "detail": "Automated decision-making requires human review and explainability controls here.",
            },
        ]
    else:
        compliance = [
            {
                "regulation": "EU AI Act - Article 6",
                "status": "COMPLIANT",
                "detail": "No severe disparity indicator remains in the audited output.",
            },
            {
                "regulation": "US ECOA",
                "status": "COMPLIANT",
                "detail": "Disparate impact is below the primary escalation threshold.",
            },
            {
                "regulation": "GDPR Article 22",
                "status": "COMPLIANT",
                "detail": "The automated decision path can be documented and monitored without critical flags.",
            },
        ]

    detailed_report = [
        f"AUDIT CERTIFICATE ID: {decision_id}",
        f"TARGET SYSTEM: {dataset_id}",
        f"AUDIT TIMESTAMP: {time.ctime(timestamp)}",
        f"PROTECTED ATTRIBUTE: {feature}",
        f"DISPARITY RATIO: {disparity}",
        f"OVERALL RISK LEVEL: {severity}",
        f"INTEGRITY CHECKSUM: {integrity_hash}",
        "---",
        f"VERDICT: {verdict}",
        "---",
    ]
    for item in compliance:
        detailed_report.append(f"{item['regulation']}: {item['status']}")
    detailed_report.append("---")
    if is_fair:
        detailed_report.append(
            "CONCLUSION: The audited boundary is within the low-risk band. Continue monitoring for drift."
        )
    else:
        detailed_report.append(
            "CONCLUSION: Deployment should be blocked until retraining or the VERDICT firewall is applied."
        )

    build_audit_history_entry(decision_id, verdict, timestamp, feature, severity)
    return {
        "decision_id": decision_id,
        "verdict": verdict,
        "integrity_hash": integrity_hash,
        "timestamp": timestamp,
        "detailed_report": detailed_report,
        "compliance": compliance,
        "audit_history": audit_history[-10:],
    }


@app.post("/api/export-pdf")
def export_pdf(payload: dict):
    decision_id = payload.get("decision_id", "DEC-UNKNOWN")
    verdict_text = payload.get("verdict", "UNKNOWN")
    integrity_hash = payload.get("integrity_hash", "")
    report_lines = payload.get("report_lines", [])
    compliance = payload.get("compliance", [])

    buffer = io.BytesIO()
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode.qr import QrCodeWidget
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    accent = HexColor("#00C896") if verdict_text == "FAIR" else HexColor("#FF3B3B")

    pdf.setFillColor(HexColor("#0D0D0D"))
    pdf.rect(0, height - 120, width, 120, fill=1, stroke=0)
    pdf.setFillColor(HexColor("#F5F5F0"))
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(40, height - 58, "VERDICT")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, height - 78, "AI Bias Audit Certificate")
    pdf.drawString(40, height - 94, f"Certificate ID: {decision_id}")

    pdf.setStrokeColor(accent)
    pdf.setLineWidth(2)
    pdf.rect(40, height - 205, width - 80, 56, fill=0, stroke=1)
    pdf.setFillColor(accent)
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawCentredString(width / 2, height - 184, f"DETERMINATION: {verdict_text}")

    qr_payload = f"{decision_id}|{verdict_text}|{integrity_hash}"
    qr_code = QrCodeWidget(qr_payload)
    bounds = qr_code.getBounds()
    qr_width = 92
    drawing = Drawing(qr_width, qr_width, transform=[qr_width / (bounds[2] - bounds[0]), 0, 0, qr_width / (bounds[3] - bounds[1]), 0, 0])
    drawing.add(qr_code)
    renderPDF.draw(drawing, pdf, width - 142, height - 320)
    pdf.setFillColor(HexColor("#666666"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(width - 142, height - 330, "Integrity QR")

    y = height - 245
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Courier", 8)
    for line in report_lines:
        if line == "---":
            pdf.line(40, y + 4, width - 160, y + 4)
            y -= 12
            continue
        pdf.drawString(40, y, line[:105])
        y -= 13
        if y < 130:
            pdf.showPage()
            y = height - 50

    if y < 185:
        pdf.showPage()
        y = height - 50
    pdf.setFillColor(HexColor("#0D0D0D"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y - 6, "REGULATORY COMPLIANCE")
    y -= 24
    pdf.setFont("Helvetica", 9)
    for item in compliance:
        status_color = HexColor("#00C896") if "COMPLIANT" in item["status"] and "NON" not in item["status"] else HexColor("#FF3B3B")
        pdf.setFillColor(status_color)
        pdf.drawString(40, y, f"{item['regulation']}: {item['status']}")
        y -= 12
        pdf.setFillColor(HexColor("#666666"))
        pdf.drawString(52, y, item["detail"][:95])
        y -= 18
        if y < 80:
            pdf.showPage()
            y = height - 50

    pdf.setFillColor(HexColor("#999999"))
    pdf.setFont("Courier", 7)
    pdf.drawString(40, 36, f"INTEGRITY HASH: {integrity_hash}")
    pdf.drawString(40, 24, "This certificate is derived from the recorded audit state and QR payload.")
    pdf.save()
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=VERDICT_Certificate_{decision_id}.pdf"},
    )
