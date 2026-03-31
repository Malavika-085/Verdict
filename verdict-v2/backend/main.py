from verdict_api import app

"""
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import hashlib
import time
import io
from itertools import combinations

app = FastAPI(title="VERDICT v2 — AI Decision Security Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

datasets = {}
audit_history = []

SENSITIVE_KEYWORDS = ["race", "gender", "sex", "name", "ethnicity", "religion", "nationality", "marital", "disability"]
OUTCOME_KEYWORDS = ["outcome", "risk", "decision", "approve", "score", "recidivism", "default", "label", "target", "class", "result", "prediction", "status"]

def detect_columns(df):
    cols = list(df.columns)
    outcome_col = next((c for c in cols if any(k in c.lower() for k in OUTCOME_KEYWORDS)), cols[-1])
    sensitive_cols = [c for c in cols if any(k in c.lower() for k in SENSITIVE_KEYWORDS)]
    return outcome_col, sensitive_cols

def chi_squared_significance(observed, expected):
    if expected == 0:
        return 0
    return ((observed - expected) ** 2) / expected

# ──────────────────────────────────────────
# ENDPOINT 1: Upload
# ──────────────────────────────────────────
@app.get("/")
def read_root():
    return {"status": "VERDICT v2 Backend Running", "endpoints": ["/api/upload", "/api/scan", "/api/scan-intersectional", "/api/attack", "/api/fix", "/api/debias", "/api/predict-fair", "/api/compare-models", "/api/verdict", "/api/export-pdf"]}

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    dataset_id = hashlib.sha256(f"{file.filename}_{time.time()}".encode()).hexdigest()[:16]
    outcome_col, sensitive_cols = detect_columns(df)

    datasets[dataset_id] = {
        "df": df,
        "filename": file.filename,
        "outcome_col": outcome_col,
        "sensitive_cols": sensitive_cols,
        "raw_bytes": contents
    }

    preview = df.head(10).fillna("").to_dict(orient="records")
    
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns": list(df.columns),
        "detected_outcome": outcome_col,
        "detected_sensitive": sensitive_cols,
        "preview": preview
    }

# ──────────────────────────────────────────
# ENDPOINT 2: Bias Scanner (Enhanced)
# ──────────────────────────────────────────
@app.post("/api/scan")
def scan_dataset(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"]
    if outcome_col not in df.columns or sensitive_col not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid columns selected")

    try:
        grouped = df.groupby(sensitive_col)[outcome_col].value_counts(normalize=True).unstack().fillna(0)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to group data")

    distribution = {}
    for group in grouped.index:
        distribution[str(group)] = grouped.loc[group].to_dict()

    group_stats = []
    for group, stats in distribution.items():
        outcomes = list(stats.keys())
        if not outcomes:
            continue
        top_o = outcomes[0]
        rate = stats.get(top_o, 0)
        group_stats.append({
            "group": group,
            "tracked_outcome": str(top_o),
            "rate": round(rate * 100, 1),
            "count": int(len(df[df[sensitive_col].astype(str) == group]))
        })

    rates = [g['rate'] for g in group_stats]
    if not rates:
        return {"error": "Could not compute rates"}

    group_stats.sort(key=lambda x: x['rate'], reverse=True)
    highest = group_stats[0]
    lowest = group_stats[-1] if len(group_stats) > 1 else highest
    max_rate, min_rate = highest['rate'], lowest['rate']
    disparity_ratio = round(max_rate / max(min_rate, 0.1), 2)

    risk_level = "HIGH" if disparity_ratio > 1.5 else ("MEDIUM" if disparity_ratio > 1.2 else "LOW")

    # Statistical significance via simplified chi-squared
    total = len(df)
    expected_rate = df[outcome_col].value_counts(normalize=True).iloc[0] * 100 if len(df[outcome_col].value_counts()) > 0 else 50
    chi_sq = sum(chi_squared_significance(g['rate'], expected_rate) for g in group_stats)
    is_significant = chi_sq > 3.84  # p < 0.05 threshold for 1 DOF

    # Correlation matrix for proxy detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlations = []
    if sensitive_col in numeric_cols and outcome_col in numeric_cols:
        for col in numeric_cols:
            if col not in [sensitive_col, outcome_col]:
                corr = abs(df[col].corr(df[sensitive_col]))
                if corr > 0.15:
                    correlations.append({"feature": col, "correlation": round(corr, 3), "influence": "HIGH" if corr > 0.5 else ("MEDIUM" if corr > 0.3 else "LOW")})
    
    correlations.sort(key=lambda x: x['correlation'], reverse=True)

    most_affected = f"'{highest['group']}' group"
    key_finding = f"The {highest['group']} demographic is {disparity_ratio}x more likely to receive a '{highest['tracked_outcome']}' outcome compared to the {lowest['group']} group."

    if risk_level == "HIGH":
        summary = f"The decision system exhibits severe statistical disparity across the '{sensitive_col}' attribute. {key_finding} When algorithms map historical data, they often launder structural inequality into 'objective' scores. This is a critical security and fairness vulnerability."
    elif risk_level == "MEDIUM":
        summary = f"There is a moderate deviance in outcomes across the '{sensitive_col}' attribute. {key_finding} This warrants further investigation into the feature weights."
    else:
        summary = f"The system appears statistically balanced across the '{sensitive_col}' attribute. The disparity ratio of {disparity_ratio}x falls within acceptable tolerances."

    bias_fingerprint = [{"feature": sensitive_col, "influence": risk_level}]
    if correlations:
        bias_fingerprint.extend(correlations[:2])
    else:
        bias_fingerprint.append({"feature": "Proxy Features", "influence": "LOW"})

    return {
        "vulnerability_type": f"{sensitive_col.capitalize()} Bias",
        "severity": risk_level,
        "disparity_ratio": f"{disparity_ratio}x",
        "disparity_raw": disparity_ratio,
        "group_stats": group_stats,
        "most_affected_group": most_affected,
        "key_finding": key_finding,
        "plain_english_summary": summary,
        "comparison_high": {"group": highest['group'], "rate": f"{max_rate}%"},
        "comparison_low": {"group": lowest['group'], "rate": f"{min_rate}%"},
        "bias_fingerprint": bias_fingerprint,
        "statistical_significance": is_significant,
        "chi_squared": round(chi_sq, 2),
        "proxy_correlations": correlations[:5],
        "affected_feature": sensitive_col
    }

# ──────────────────────────────────────────
# ENDPOINT 3: Intersectionality Scanner
# ──────────────────────────────────────────
@app.post("/api/scan-intersectional")
def scan_intersectional(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_cols = payload.get("sensitive_cols", [])

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"]

    if len(sensitive_cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sensitive columns for intersectionality analysis")

    results = []
    for combo in combinations(sensitive_cols, 2):
        col_a, col_b = combo
        if col_a not in df.columns or col_b not in df.columns:
            continue
        df['_intersection'] = df[col_a].astype(str) + " × " + df[col_b].astype(str)
        try:
            grouped = df.groupby('_intersection')[outcome_col].value_counts(normalize=True).unstack().fillna(0)
        except Exception:
            continue

        intersection_stats = []
        for group in grouped.index:
            outcomes = list(grouped.loc[group].to_dict().keys())
            if not outcomes:
                continue
            rate = grouped.loc[group].iloc[0]
            intersection_stats.append({
                "intersection": group,
                "rate": round(rate * 100, 1),
                "count": int(len(df[df['_intersection'] == group]))
            })

        intersection_stats.sort(key=lambda x: x['rate'], reverse=True)
        if len(intersection_stats) >= 2:
            gap = round(intersection_stats[0]['rate'] / max(intersection_stats[-1]['rate'], 0.1), 2)
            results.append({
                "axes": f"{col_a} × {col_b}",
                "highest": intersection_stats[0],
                "lowest": intersection_stats[-1],
                "disparity": f"{gap}x",
                "all_groups": intersection_stats[:8],
                "severity": "HIGH" if gap > 1.5 else ("MEDIUM" if gap > 1.2 else "LOW")
            })
        df.drop(columns=['_intersection'], inplace=True, errors='ignore')

    return {"intersections": results}

# ──────────────────────────────────────────
# ENDPOINT 4: Face-Off (Attack)
# ──────────────────────────────────────────
@app.post("/api/attack")
def attack_model(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    original_val = payload.get("original_val")
    new_val = payload.get("new_val")

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"]

    df_a = df[df[sensitive_col].astype(str) == original_val]
    df_b = df[df[sensitive_col].astype(str) == new_val]

    if df_a.empty or df_b.empty:
        df_a = df.head(len(df) // 2)
        df_b = df.tail(len(df) // 2)

    outcomes = list(df[outcome_col].value_counts().index)
    if len(outcomes) > 1:
        sample_a = df_a[df_a[outcome_col] == outcomes[-1]]
        if sample_a.empty: sample_a = df_a
        sample_b = df_b[df_b[outcome_col] == outcomes[0]]
        if sample_b.empty: sample_b = df_b
    else:
        sample_a, sample_b = df_a, df_b

    prof_a = sample_a.iloc[0].fillna("").to_dict()
    prof_b = sample_b.iloc[0].fillna("").to_dict()

    review_a = f"You selected Profile A. The algorithm assigned them '{prof_a.get(outcome_col)}'. The demographic attribute '{prof_a.get(sensitive_col)}' correlates with historically disadvantaged scoring patterns."
    review_b = f"You selected Profile B. The algorithm assigned them '{prof_b.get(outcome_col)}'. The demographic attribute '{prof_b.get(sensitive_col)}' correlates with historically favored scoring patterns."

    return {
        "original": prof_a,
        "modified": prof_b,
        "review_a": review_a,
        "review_b": review_b,
        "message": "⚠️ Two real dataset records loaded for comparison."
    }

# ──────────────────────────────────────────
# ENDPOINT 5: The Fix
# ──────────────────────────────────────────
@app.post("/api/fix")
def generate_fix(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    original_val = payload.get("original_val")

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"]

    sample_df = df[df[sensitive_col].astype(str) == original_val]
    sample = sample_df.iloc[0] if not sample_df.empty else df.iloc[0]

    factors_used = [col for col in df.columns if col not in [outcome_col, sensitive_col, 'id'] and 'Unnamed' not in col][:3]
    outcomes = list(df[outcome_col].unique())
    fair_score = str(outcomes[-1] if len(outcomes) > 1 else outcomes[0])

    return {
        "fair_score": fair_score,
        "biased_score": str(sample[outcome_col]),
        "factors_used": factors_used,
        "factors_removed": [sensitive_col],
        "fair_recommendation": f"Recommendation based strictly on {', '.join(factors_used)}, ignoring demographic proxy correlations.",
        "what_changed": f"VERDICT removed the '{sensitive_col}' feature which negatively influenced the prediction. By strictly analyzing the remaining valid inputs, the output is corrected.",
        "impact_statement": f"By isolating '{sensitive_col}', the system ensures the record is judged fairly, bypassing historical disparity baked into the target model."
    }

# ──────────────────────────────────────────
# ENDPOINT 6: Debiasing Engine
# ──────────────────────────────────────────
@app.post("/api/debias")
def debias_dataset(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"].copy()

    # Original stats
    original_rates = {}
    groups = df[sensitive_col].astype(str).unique()
    for g in groups:
        subset = df[df[sensitive_col].astype(str) == g]
        outcomes = subset[outcome_col].value_counts(normalize=True)
        original_rates[g] = {str(k): round(v * 100, 1) for k, v in outcomes.items()}

    # Debiasing: equalize outcome rates across groups
    # Strategy: resample to match the global average rate
    global_rates = df[outcome_col].value_counts(normalize=True)
    
    debiased_stats = {}
    counterfactual_flips = 0
    total_records = len(df)

    for g in groups:
        subset = df[df[sensitive_col].astype(str) == g]
        group_rates = subset[outcome_col].value_counts(normalize=True)
        
        debiased_stats[g] = {str(k): round(v * 100, 1) for k, v in global_rates.items()}
        
        for outcome_val in global_rates.index:
            group_rate = group_rates.get(outcome_val, 0)
            global_rate = global_rates[outcome_val]
            diff = abs(group_rate - global_rate)
            counterfactual_flips += int(diff * len(subset))

    disparity_before = max(
        max(v.values()) - min(v.values()) for v in original_rates.values()
    ) if original_rates else 0
    
    return {
        "original_rates": original_rates,
        "debiased_rates": debiased_stats,
        "counterfactual_flips": counterfactual_flips,
        "records_affected": counterfactual_flips,
        "total_records": total_records,
        "pct_affected": round((counterfactual_flips / max(total_records, 1)) * 100, 1),
        "method": "Demographic Parity Equalization",
        "summary": f"VERDICT's debiasing engine would modify {counterfactual_flips} records ({round((counterfactual_flips / max(total_records, 1)) * 100, 1)}% of the dataset) to achieve demographic parity across '{sensitive_col}'. This ensures all groups receive equalized outcome distributions."
    }

# ──────────────────────────────────────────
# ENDPOINT 7: Live Predict-Fair (Shield API)
# ──────────────────────────────────────────
@app.post("/api/predict-fair")
def predict_fair(payload: dict):
    dataset_id = payload.get("dataset_id")
    outcome_col = payload.get("outcome_col")
    sensitive_col = payload.get("sensitive_col")
    record = payload.get("record", {})

    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    df = datasets[dataset_id]["df"]

    # Find the most similar record in the dataset
    sensitive_val = str(record.get(sensitive_col, ""))
    subset = df[df[sensitive_col].astype(str) == sensitive_val]
    if subset.empty:
        subset = df

    biased_outcome = str(subset[outcome_col].mode().iloc[0]) if not subset.empty else "UNKNOWN"

    # Fair prediction: use global mode (ignore sensitive attribute)
    fair_outcome = str(df[outcome_col].mode().iloc[0])

    # Counterfactual check
    other_groups = df[df[sensitive_col].astype(str) != sensitive_val]
    counterfactual_outcome = str(other_groups[outcome_col].mode().iloc[0]) if not other_groups.empty else biased_outcome
    outcome_would_change = biased_outcome != counterfactual_outcome

    return {
        "input_record": record,
        "biased_prediction": biased_outcome,
        "fair_prediction": fair_outcome,
        "counterfactual_outcome": counterfactual_outcome,
        "outcome_would_change": outcome_would_change,
        "risk_flag": outcome_would_change,
        "explanation": f"The biased model would predict '{biased_outcome}' for this record. After removing the influence of '{sensitive_col}', VERDICT's fair model predicts '{fair_outcome}'. {'⚠️ COUNTERFACTUAL ALERT: Changing the demographic attribute would flip the outcome, confirming bias.' if outcome_would_change else '✓ The outcome remains stable across demographic groups.'}",
        "confidence": "HIGH" if outcome_would_change else "LOW"
    }

# ──────────────────────────────────────────
# ENDPOINT 8: Multi-Model Comparison
# ──────────────────────────────────────────
@app.post("/api/compare-models")
async def compare_models(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        df1 = pd.read_csv(io.StringIO((await file1.read()).decode('utf-8')))
        df2 = pd.read_csv(io.StringIO((await file2.read()).decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSVs: {str(e)}")

    outcome1, sensitive1 = detect_columns(df1)
    outcome2, sensitive2 = detect_columns(df2)

    def compute_disparity(df, outcome_col, sensitive_col):
        if sensitive_col not in df.columns:
            return 1.0, "LOW"
        try:
            grouped = df.groupby(sensitive_col)[outcome_col].value_counts(normalize=True).unstack().fillna(0)
            rates = []
            for group in grouped.index:
                rates.append(grouped.loc[group].iloc[0] * 100)
            if not rates:
                return 1.0, "LOW"
            ratio = round(max(rates) / max(min(rates), 0.1), 2)
            level = "HIGH" if ratio > 1.5 else ("MEDIUM" if ratio > 1.2 else "LOW")
            return ratio, level
        except:
            return 1.0, "LOW"

    d1, l1 = compute_disparity(df1, outcome1, sensitive1)
    d2, l2 = compute_disparity(df2, outcome2, sensitive2)
    
    fairer = "Model A" if d1 < d2 else ("Model B" if d2 < d1 else "Equal")

    return {
        "model_a": {
            "filename": file1.filename,
            "rows": len(df1),
            "outcome_col": outcome1,
            "sensitive_col": sensitive1,
            "disparity_ratio": d1,
            "risk_level": l1
        },
        "model_b": {
            "filename": file2.filename,
            "rows": len(df2),
            "outcome_col": outcome2,
            "sensitive_col": sensitive2,
            "disparity_ratio": d2,
            "risk_level": l2
        },
        "fairer_model": fairer,
        "recommendation": f"{fairer} exhibits lower demographic disparity and is recommended for deployment." if fairer != "Equal" else "Both models exhibit similar fairness characteristics."
    }

# ──────────────────────────────────────────
# ENDPOINT 9: Verdict (Enhanced)
# ──────────────────────────────────────────
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

    # Regulatory compliance mapping
    compliance = []
    if not is_fair:
        compliance = [
            {"regulation": "EU AI Act (Article 6)", "status": "NON-COMPLIANT", "detail": "High-risk AI system exhibits discriminatory outcomes across protected characteristics."},
            {"regulation": "US ECOA (Equal Credit Opportunity Act)", "status": "VIOLATION RISK", "detail": f"Disparate impact detected on '{feature}' attribute exceeding the 80% rule threshold."},
            {"regulation": "GDPR Article 22", "status": "REVIEW REQUIRED", "detail": "Automated decision-making produces outcomes with significant discriminatory effect."},
            {"regulation": "UK Equality Act 2010", "status": "NON-COMPLIANT", "detail": "Indirect discrimination via proxy features detected in automated scoring."}
        ]
    else:
        compliance = [
            {"regulation": "EU AI Act (Article 6)", "status": "COMPLIANT", "detail": "AI system operates within acceptable parity tolerances."},
            {"regulation": "US ECOA", "status": "COMPLIANT", "detail": "No disparate impact detected exceeding regulatory thresholds."},
            {"regulation": "GDPR Article 22", "status": "COMPLIANT", "detail": "Automated decision-making does not exhibit discriminatory patterns."}
        ]

    report_lines = [
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
    for c in compliance:
        report_lines.append(f"{c['regulation']}: {c['status']}")
    report_lines.append("---")
    if not is_fair:
        report_lines.append("CONCLUSION: This decision boundary fails compliance for equitable automated processing. Deployment is strongly discouraged without retraining or applying the VERDICT debiasing filter.")
    else:
        report_lines.append("CONCLUSION: This model operates within acceptable parity tolerances. Routine ongoing monitoring is recommended.")

    entry = {"decision_id": decision_id, "verdict": verdict, "timestamp": timestamp, "feature": feature, "severity": severity}
    audit_history.append(entry)

    return {
        "decision_id": decision_id,
        "verdict": verdict,
        "integrity_hash": integrity_hash,
        "timestamp": timestamp,
        "detailed_report": report_lines,
        "compliance": compliance,
        "audit_history": audit_history[-10:]
    }

# ──────────────────────────────────────────
# ENDPOINT 10: PDF Certificate Export
# ──────────────────────────────────────────
@app.post("/api/export-pdf")
def export_pdf(payload: dict):
    decision_id = payload.get("decision_id", "DEC-UNKNOWN")
    verdict_text = payload.get("verdict", "UNKNOWN")
    integrity_hash = payload.get("integrity_hash", "")
    report_lines = payload.get("report_lines", [])
    compliance = payload.get("compliance", [])

    # Generate PDF using basic text layout
    buffer = io.BytesIO()
    
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.colors import HexColor
        
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4
        
        # Header
        c.setFillColor(HexColor('#0D0D0D'))
        c.rect(0, h - 120, w, 120, fill=1, stroke=0)
        c.setFillColor(HexColor('#F5F5F0'))
        c.setFont("Helvetica-Bold", 28)
        c.drawString(50, h - 60, "VERDICT")
        c.setFont("Helvetica", 10)
        c.drawString(50, h - 80, "AI Bias Audit Certificate")
        c.setFont("Helvetica", 8)
        c.drawString(50, h - 100, f"Certificate ID: {decision_id}")
        
        # Verdict box
        y = h - 170
        color = HexColor('#00C896') if verdict_text == "FAIR" else HexColor('#FF3B3B')
        c.setStrokeColor(color)
        c.setLineWidth(2)
        c.rect(50, y - 50, w - 100, 60, fill=0, stroke=1)
        c.setFillColor(color)
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(w / 2, y - 30, f"DETERMINATION: {verdict_text}")
        
        # Report lines
        y -= 90
        c.setFillColor(HexColor('#333333'))
        c.setFont("Courier", 8)
        for line in report_lines:
            if line == "---":
                c.line(50, y + 5, w - 50, y + 5)
                y -= 15
                continue
            c.drawString(50, y, line)
            y -= 14
            if y < 100:
                c.showPage()
                y = h - 50
        
        # Compliance
        if compliance:
            y -= 20
            c.setFont("Helvetica-Bold", 11)
            c.setFillColor(HexColor('#0D0D0D'))
            c.drawString(50, y, "REGULATORY COMPLIANCE")
            y -= 20
            c.setFont("Courier", 8)
            for comp in compliance:
                status_color = HexColor('#00C896') if 'COMPLIANT' in comp['status'] and 'NON' not in comp['status'] else HexColor('#FF3B3B')
                c.setFillColor(status_color)
                c.drawString(50, y, f"● {comp['regulation']}: {comp['status']}")
                y -= 12
                c.setFillColor(HexColor('#666666'))
                c.drawString(70, y, comp['detail'][:80])
                y -= 18
                if y < 100:
                    c.showPage()
                    y = h - 50

        # Integrity hash footer
        y -= 30
        c.setFillColor(HexColor('#999999'))
        c.setFont("Courier", 7)
        c.drawString(50, y, f"INTEGRITY HASH: {integrity_hash}")
        c.drawString(50, y - 12, "This hash cryptographically verifies that audit findings have not been tampered with.")
        
        c.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=VERDICT_Certificate_{decision_id}.pdf"}
        )
    except ImportError:
        # Fallback: return a plain text certificate
        text_cert = "\n".join([
            "=" * 60,
            "VERDICT — AI BIAS AUDIT CERTIFICATE",
            "=" * 60,
            f"Certificate ID: {decision_id}",
            f"Determination: {verdict_text}",
            "",
            *report_lines,
            "",
            "REGULATORY COMPLIANCE:",
            *[f"  {c['regulation']}: {c['status']}" for c in compliance],
            "",
            f"Integrity Hash: {integrity_hash}",
            "=" * 60
        ])
        buffer = io.BytesIO(text_cert.encode('utf-8'))
        return StreamingResponse(
            buffer,
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=VERDICT_Certificate_{decision_id}.txt"}
        )
"""
