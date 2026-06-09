import os
import json
from datetime import datetime

# ═════════════════════════════════════════════════════════════
#  STATIC CLINICAL DATA
# ═════════════════════════════════════════════════════════════
CLASSES = ["Normal", "Crackle", "Wheeze", "Both"]

CLINICAL_INFO = {
    "Normal": {
        "urgency":     "none",
        "color":       "#00ffa3",
        "summary":     "No abnormal respiratory sounds detected.",
        "findings":    ("The lung sound analysis did not detect any crackles or "
                        "wheezes. Airflow patterns appear within normal range."),
        "conditions":  [],
        "advice":      ("No immediate medical attention required. Maintain a healthy "
                        "lifestyle, avoid smoking, and monitor for any new symptoms."),
        "next_steps":  [
            "Continue regular health monitoring",
            "Return if new symptoms develop",
            "Annual check-up recommended"
        ],
        "doctor_note": ("AI screening result: Normal. No further respiratory "
                        "investigation indicated unless clinically warranted.")
    },
    "Crackle": {
        "urgency":     "moderate",
        "color":       "#ffd166",
        "summary":     "Crackle sounds detected in lung audio.",
        "findings":    ("Discontinuous crackling or popping sounds were identified "
                        "during inhalation. This pattern is associated with fluid or "
                        "secretions in the airways."),
        "conditions":  ["Pneumonia", "Bronchitis", "Heart failure", "Pulmonary fibrosis"],
        "advice":      ("Please consult a doctor within the next few days. "
                        "Bring this AI report to your appointment."),
        "next_steps":  [
            "Visit a GP or pulmonologist",
            "Chest X-ray may be recommended",
            "Blood tests to rule out infection",
            "Sputum culture if productive cough"
        ],
        "doctor_note": ("AI screening result: Crackles detected. Recommend clinical "
                        "examination, chest auscultation, and chest X-ray to rule out "
                        "pneumonia or heart failure.")
    },
    "Wheeze": {
        "urgency":     "high",
        "color":       "#ff8c42",
        "summary":     "Wheeze sounds detected in lung audio.",
        "findings":    ("Continuous high-pitched whistling sounds were identified "
                        "during breathing. This pattern indicates narrowed or "
                        "obstructed airways."),
        "conditions":  ["Asthma", "COPD", "Bronchospasm", "Allergic reaction"],
        "advice":      ("Please seek medical advice soon. If you experience difficulty "
                        "breathing, seek emergency care immediately."),
        "next_steps":  [
            "See a doctor as soon as possible",
            "Spirometry (lung function test)",
            "Bronchodilator trial",
            "Allergy testing if suspected"
        ],
        "doctor_note": ("AI screening result: Wheeze detected. Recommend spirometry, "
                        "bronchodilator response test, and clinical assessment for "
                        "asthma or COPD.")
    },
    "Both": {
        "urgency":     "critical",
        "color":       "#ff4d6d",
        "summary":     "Both crackles and wheezes detected.",
        "findings":    ("The analysis identified both discontinuous crackle sounds and "
                        "continuous wheeze sounds. This combination may indicate a "
                        "complex or severe respiratory condition."),
        "conditions":  ["Severe COPD", "Mixed respiratory infection",
                        "Acute respiratory failure", "Bronchiectasis"],
        "advice":      ("Please see a doctor immediately. This combination of findings "
                        "requires urgent medical evaluation."),
        "next_steps":  [
            "Seek urgent medical attention",
            "Emergency assessment if breathless",
            "CT chest scan",
            "Full pulmonary function testing"
        ],
        "doctor_note": ("AI screening result: Both crackles and wheezes detected. "
                        "Urgent clinical evaluation recommended. Consider CT chest, "
                        "ABG, and specialist referral.")
    },
    "Uncertain": {
        "urgency":     "review",
        "color":       "#888780",
        "summary":     "Prediction confidence too low for a reliable result.",
        "findings":    ("The model was unable to identify a clear pattern in the audio. "
                        "This may be due to background noise, poor microphone placement, "
                        "or an unusual sound pattern."),
        "conditions":  [],
        "advice":      ("Please re-record in a quieter environment with the stethoscope "
                        "held firmly against the chest. If symptoms persist, consult a doctor."),
        "next_steps":  [
            "Re-record audio in a quieter room",
            "Ensure stethoscope is on bare skin",
            "Try a longer recording (5-10 seconds)",
            "Consult a doctor if symptoms are present"
        ],
        "doctor_note": ("AI screening result: Uncertain. Model confidence below threshold. "
                        "Manual auscultation required.")
    }
}


# ═════════════════════════════════════════════════════════════
#  RISK STRATIFICATION
# ═════════════════════════════════════════════════════════════
def get_risk_level(pred_class_name, confidence, patient_meta=None):
    """
    Computes a 0-100 risk score combining:
      - Predicted disease severity
      - Model confidence  (gradual adjustment, not a cliff)
      - Patient risk factors (age, smoking)
    """
    base_risk_map = {
        "Normal":    0,
        "Crackle":  40,
        "Wheeze":   65,
        "Both":     85,
        "Uncertain": 20   # low-but-nonzero: something is unclear
    }
    base_risk = base_risk_map.get(pred_class_name, 20)

    # Confidence adjustment — gradual, not binary
    if confidence < 50:
        base_risk = max(base_risk - 20, 0)
    elif confidence < 60:
        base_risk = max(base_risk - 10, 0)
    elif confidence > 90:
        base_risk = min(base_risk + 10, 100)
    elif confidence > 80:
        base_risk = min(base_risk + 5, 100)

    # Patient risk factors
    if patient_meta:
        age     = patient_meta.get("age", 40)
        smoking = patient_meta.get("smoking", False)
        copd    = patient_meta.get("copd_history", False)
        if age > 70:
            base_risk = min(base_risk + 15, 100)
        elif age > 60:
            base_risk = min(base_risk + 10, 100)
        if smoking:
            base_risk = min(base_risk + 15, 100)
        if copd:
            base_risk = min(base_risk + 10, 100)

    # Risk label
    if base_risk < 20:
        label  = "Low"
        color  = "#00ffa3"
        action = "Monitor at home"
    elif base_risk < 50:
        label  = "Moderate"
        color  = "#ffd166"
        action = "See a doctor within a week"
    elif base_risk < 75:
        label  = "High"
        color  = "#ff8c42"
        action = "See a doctor soon"
    else:
        label  = "Critical"
        color  = "#ff4d6d"
        action = "Seek urgent medical attention"

    return {
        "score":  base_risk,
        "label":  label,
        "color":  color,
        "action": action
    }


# ═════════════════════════════════════════════════════════════
#  GENERATE REPORT  (main function)
# ═════════════════════════════════════════════════════════════
def generate_clinical_report(
    pred_class,
    confidence,
    all_probs,
    model_used,
    time_taken,
    top_features=None,
    highlight_info=None,
    patient_meta=None
):
    """
    Generates a complete clinical report as a dictionary.

    Args:
        pred_class    : int 0-3  OR  str class name  (e.g. 'Wheeze')
        confidence    : float — confidence percentage (0-100)
        all_probs     : dict  — {class_name: probability_percent}
        model_used    : str   — description of model(s) used
        time_taken    : float — seconds for prediction
        top_features  : list  — [{"feature": str, "importance": float}, ...]
        highlight_info: dict  — {"peak_time_sec": float, "start_sec": float, "end_sec": float}
        patient_meta  : dict  — {"age": int, "smoking": bool, "copd_history": bool}

    Returns:
        report: dict — full clinical report
    """
    # Accept both int index and string class name
    if isinstance(pred_class, int):
        pred_class_name = CLASSES[pred_class] if 0 <= pred_class < len(CLASSES) else "Uncertain"
    else:
        pred_class_name = pred_class if pred_class in CLINICAL_INFO else "Uncertain"

    info      = CLINICAL_INFO[pred_class_name]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = {
        "timestamp":  timestamp,
        "tool_name":  "LungSense AI",
        "disclaimer": ("This is an AI-assisted screening tool for educational purposes "
                       "only. It is NOT a substitute for professional medical diagnosis."),

        # ── Prediction ──────────────────────────────────────
        "prediction": {
            "disease":    pred_class_name,
            "confidence": confidence,
            "urgency":    info["urgency"],
            "color":      info["color"],
            "all_probs":  all_probs
        },

        # ── Clinical findings ────────────────────────────────
        "findings": {
            "summary":    info["summary"],
            "detail":     info["findings"],
            "conditions": info["conditions"]
        },

        # ── Patient advice ───────────────────────────────────
        "advice": {
            "patient_message": info["advice"],
            "next_steps":      info["next_steps"]
        },

        # ── Doctor note ──────────────────────────────────────
        "doctor_note": info["doctor_note"],

        # ── AI explanation ───────────────────────────────────
        "explanation": {
            "model_used":      model_used,
            "time_taken_sec":  round(time_taken, 2),
            "top_features":    top_features or [],
            "audio_highlight": highlight_info or {}
        },

        # ── Risk stratification ──────────────────────────────
        "risk": get_risk_level(pred_class_name, confidence, patient_meta)
    }

    return report


# ═════════════════════════════════════════════════════════════
#  PLAIN TEXT FORMAT
# ═════════════════════════════════════════════════════════════
def format_report_text(report):
    """
    Converts the report dict to a readable plain-text string
    suitable for printing or saving as a .txt file.
    """
    r    = report
    p    = r["prediction"]
    f    = r["findings"]
    a    = r["advice"]
    exp  = r["explanation"]
    risk = r["risk"]

    lines = [
        "=" * 60,
        "  LUNGSENSE AI — SCREENING REPORT",
        "=" * 60,
        f"  Date     : {r['timestamp']}",
        f"  Model    : {exp['model_used']}",
        f"  Duration : {exp['time_taken_sec']}s",
        "",
        "  PREDICTION",
        "  " + "-" * 40,
        f"  Disease    : {p['disease']}",
        f"  Confidence : {p['confidence']}%",
        f"  Urgency    : {p['urgency'].upper()}",
        "",
        "  All class probabilities:",
    ]

    for cls, prob in sorted(p["all_probs"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob // 5)
        lines.append(f"    {cls:<12} {prob:>5.1f}%  {bar}")

    lines += [
        "",
        "  CLINICAL FINDINGS",
        "  " + "-" * 40,
        f"  {f['summary']}",
        f"  {f['detail']}",
        "",
        "  Possible conditions: " + (", ".join(f["conditions"]) or "None"),
        "",
        "  RISK ASSESSMENT",
        "  " + "-" * 40,
        f"  Risk level : {risk['label']} ({risk['score']}/100)",
        f"  Action     : {risk['action']}",
        "",
        "  ADVICE FOR PATIENT",
        "  " + "-" * 40,
        f"  {a['patient_message']}",
        "",
        "  Next steps:",
    ]

    for step in a["next_steps"]:
        lines.append(f"    • {step}")

    if exp["top_features"]:
        lines += [
            "",
            "  TOP PREDICTIVE FEATURES (SHAP / importance)",
            "  " + "-" * 40
        ]
        for i, feat in enumerate(exp["top_features"][:5], 1):
            lines.append(f"    {i}. {feat['feature']:<25} {feat['importance']:.4f}")

    if exp["audio_highlight"]:
        hl = exp["audio_highlight"]
        lines += [
            "",
            "  AUDIO HIGHLIGHT",
            "  " + "-" * 40,
            f"  Peak at {hl.get('peak_time_sec', '?')}s  "
            f"({hl.get('start_sec', '?')}s – {hl.get('end_sec', '?')}s)"
        ]

    lines += [
        "",
        "  NOTE FOR DOCTOR",
        "  " + "-" * 40,
        f"  {r['doctor_note']}",
        "",
        "  " + "-" * 60,
        "  DISCLAIMER",
        f"  {r['disclaimer']}",
        "=" * 60,
    ]

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════
#  HTML FORMAT  (for PDF conversion or browser display)
# ═════════════════════════════════════════════════════════════
def format_report_html(report):
    """
    Returns a self-contained HTML string of the clinical report.
    Suitable for display in a browser, sending by email, or
    converting to PDF with pdfkit / weasyprint.
    """
    r     = report
    p     = r["prediction"]
    f     = r["findings"]
    a     = r["advice"]
    exp   = r["explanation"]
    risk  = r["risk"]
    color = p["color"]

    # Build probability bars
    prob_rows = ""
    for cls, prob in sorted(p["all_probs"].items(),
                              key=lambda x: x[1], reverse=True):
        bar_color = color if cls == p["disease"] else "#444"
        prob_rows += f"""
        <tr>
          <td style="padding:4px 8px;width:90px">{cls}</td>
          <td style="padding:4px 8px">
            <div style="background:#333;border-radius:4px;height:14px;width:100%">
              <div style="background:{bar_color};width:{min(prob,100):.0f}%;
                          height:14px;border-radius:4px;"></div>
            </div>
          </td>
          <td style="padding:4px 8px;width:55px;text-align:right">{prob:.1f}%</td>
        </tr>"""

    # Next steps list
    steps_html = "".join(f"<li>{s}</li>" for s in a["next_steps"])

    # Conditions
    cond_html = (", ".join(f["conditions"])
                 if f["conditions"] else "None identified")

    # Features
    feat_html = ""
    if exp["top_features"]:
        feat_rows = "".join(
            f"<tr><td>{i}. {ft['feature']}</td>"
            f"<td style='text-align:right'>{ft['importance']:.4f}</td></tr>"
            for i, ft in enumerate(exp["top_features"][:5], 1)
        )
        feat_html = f"""
        <h3>Top predictive features</h3>
        <table style="width:100%;border-collapse:collapse">
          {feat_rows}
        </table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LungSense AI — Screening Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; background:#0a1628;
           color:#e0e0e0; margin:0; padding:24px; }}
  .card {{ background:#0f1e35; border:1px solid #1e3a5f;
           border-radius:10px; padding:20px; margin-bottom:16px; }}
  h1    {{ color:#00c4ff; font-size:20px; margin:0 0 4px; }}
  h2    {{ color:#ccc; font-size:13px; font-weight:normal; margin:0 0 16px; }}
  h3    {{ color:#aaa; font-size:14px; margin:8px 0 6px; border-bottom:1px solid #1e3a5f; padding-bottom:4px; }}
  .badge{{ display:inline-block; padding:4px 12px; border-radius:20px;
           font-size:13px; font-weight:bold; color:#000; }}
  table {{ width:100%; border-collapse:collapse; }}
  td    {{ font-size:13px; color:#ccc; }}
  ul    {{ margin:6px 0; padding-left:20px; font-size:13px; color:#ccc; }}
  .disc {{ font-size:11px; color:#666; margin-top:8px; }}
</style>
</head>
<body>

<div class="card">
  <h1>LungSense AI — Screening Report</h1>
  <h2>{r['timestamp']}  ·  Model: {exp['model_used']}  ·  {exp['time_taken_sec']}s</h2>

  <h3>Prediction</h3>
  <span class="badge" style="background:{color}">{p['disease']}</span>
  &nbsp;
  <span style="font-size:13px;color:#aaa">Confidence: <strong style="color:#fff">{p['confidence']}%</strong></span>
  &nbsp;
  <span style="font-size:13px;color:#aaa">Urgency: <strong style="color:{color}">{p['urgency'].upper()}</strong></span>

  <h3>Class probabilities</h3>
  <table>{prob_rows}</table>
</div>

<div class="card">
  <h3>Clinical findings</h3>
  <p style="font-size:13px;color:#ccc;margin:0 0 8px"><strong>{f['summary']}</strong></p>
  <p style="font-size:13px;color:#aaa;margin:0 0 8px">{f['detail']}</p>
  <p style="font-size:13px;color:#aaa;margin:0">Possible conditions: {cond_html}</p>
</div>

<div class="card">
  <h3>Risk assessment</h3>
  <span class="badge" style="background:{risk['color']}">{risk['label']} — {risk['score']}/100</span>
  <p style="font-size:13px;color:#aaa;margin:8px 0 0">{risk['action']}</p>
</div>

<div class="card">
  <h3>Advice for patient</h3>
  <p style="font-size:13px;color:#ccc;margin:0 0 8px">{a['patient_message']}</p>
  <h3>Next steps</h3>
  <ul>{steps_html}</ul>
</div>

<div class="card">
  <h3>Note for doctor</h3>
  <p style="font-size:13px;color:#ccc;margin:0">{r['doctor_note']}</p>
  {feat_html}
</div>

<p class="disc">{r['disclaimer']}</p>
</body>
</html>"""

    return html


# ═════════════════════════════════════════════════════════════
#  PDF EXPORT  (ReportLab — no external binary needed)
# ═════════════════════════════════════════════════════════════
def save_report_pdf(report, output_dir="explanations"):
    """
    Saves a formatted PDF using ReportLab.
    Install with:  pip install reportlab --break-system-packages
    Falls back gracefully if reportlab is not installed.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "clinical_report.pdf")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)

        r     = report
        p     = r["prediction"]
        f     = r["findings"]
        a     = r["advice"]
        exp   = r["explanation"]
        risk  = r["risk"]

        doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                   leftMargin=20*mm, rightMargin=20*mm,
                                   topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                     fontSize=16, spaceAfter=4)
        h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"],
                                     fontSize=12, spaceAfter=4)
        body_style  = ParagraphStyle("body",  parent=styles["Normal"],
                                     fontSize=10, spaceAfter=4)
        small_style = ParagraphStyle("small", parent=styles["Normal"],
                                     fontSize=8,  textColor=colors.grey)

        elements = []

        # ── Header ──────────────────────────────────────────
        elements.append(Paragraph("LungSense AI — Screening Report", title_style))
        elements.append(Paragraph(
            f"{r['timestamp']}  ·  Model: {exp['model_used']}  ·  {exp['time_taken_sec']}s",
            small_style))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Spacer(1, 6))

        # ── Prediction ──────────────────────────────────────
        elements.append(Paragraph("Prediction", h2_style))
        pred_data = [
            ["Disease",    p["disease"]],
            ["Confidence", f"{p['confidence']}%"],
            ["Urgency",    p["urgency"].upper()],
        ]
        pred_table = Table(pred_data, colWidths=[60*mm, 110*mm])
        pred_table.setStyle(TableStyle([
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",   (0, 0), (0, -1),  colors.grey),
            ("BACKGROUND",  (0, 0), (-1, -1), colors.whitesmoke),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.whitesmoke, colors.white]),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",(0, 0), (-1, -1), 6),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0), (-1, -1), 4),
        ]))
        elements.append(pred_table)
        elements.append(Spacer(1, 8))

        # ── Probabilities ────────────────────────────────────
        elements.append(Paragraph("Class Probabilities", h2_style))
        prob_rows = [["Class", "Probability"]]
        for cls, prob in sorted(p["all_probs"].items(),
                                  key=lambda x: x[1], reverse=True):
            prob_rows.append([cls, f"{prob:.1f}%"])
        prob_table = Table(prob_rows, colWidths=[60*mm, 110*mm])
        prob_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.lightgrey),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0,0), (-1, -1), 3),
        ]))
        elements.append(prob_table)
        elements.append(Spacer(1, 8))

        # ── Findings ─────────────────────────────────────────
        elements.append(Paragraph("Clinical Findings", h2_style))
        elements.append(Paragraph(f"<b>{f['summary']}</b>", body_style))
        elements.append(Paragraph(f['detail'], body_style))
        if f["conditions"]:
            elements.append(Paragraph(
                "Possible conditions: " + ", ".join(f["conditions"]),
                body_style))
        elements.append(Spacer(1, 6))

        # ── Risk ─────────────────────────────────────────────
        elements.append(Paragraph("Risk Assessment", h2_style))
        elements.append(Paragraph(
            f"<b>{risk['label']}</b>  ({risk['score']}/100)  —  {risk['action']}",
            body_style))
        elements.append(Spacer(1, 6))

        # ── Advice ───────────────────────────────────────────
        elements.append(Paragraph("Advice for Patient", h2_style))
        elements.append(Paragraph(a["patient_message"], body_style))
        for step in a["next_steps"]:
            elements.append(Paragraph(f"• {step}", body_style))
        elements.append(Spacer(1, 6))

        # ── Doctor note ──────────────────────────────────────
        elements.append(Paragraph("Note for Doctor", h2_style))
        elements.append(Paragraph(r["doctor_note"], body_style))
        elements.append(Spacer(1, 6))

        # ── Features ─────────────────────────────────────────
        if exp["top_features"]:
            elements.append(Paragraph("Top Predictive Features", h2_style))
            feat_rows = [["#", "Feature", "Importance"]]
            for i, ft in enumerate(exp["top_features"][:5], 1):
                feat_rows.append([str(i), ft["feature"], f"{ft['importance']:.4f}"])
            feat_table = Table(feat_rows, colWidths=[15*mm, 100*mm, 55*mm])
            feat_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  colors.lightgrey),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",  (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING",(0,0), (-1, -1), 3),
            ]))
            elements.append(feat_table)
            elements.append(Spacer(1, 6))

        # ── Disclaimer ───────────────────────────────────────
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(r["disclaimer"], small_style))

        doc.build(elements)
        print(f"  Report (PDF)  saved → {pdf_path}")
        return pdf_path

    except ImportError:
        print("  ⚠ ReportLab not installed. Run: pip install reportlab --break-system-packages")
        print("  ⚠ PDF export skipped — TXT and JSON still saved.")
        return None


# ═════════════════════════════════════════════════════════════
#  SAVE ALL FORMATS
# ═════════════════════════════════════════════════════════════
def save_report(report, output_dir="explanations"):
    """
    Saves report in JSON, TXT, HTML, and PDF formats.
    Returns a dict of saved paths (None if format failed).
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    # JSON
    json_path = os.path.join(output_dir, "clinical_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report (JSON) saved → {json_path}")
    paths["json"] = json_path

    # Plain text
    txt_path = os.path.join(output_dir, "clinical_report.txt")
    with open(txt_path, "w") as f:
        f.write(format_report_text(report))
    print(f"  Report (TXT)  saved → {txt_path}")
    paths["txt"] = txt_path

    # HTML
    html_path = os.path.join(output_dir, "clinical_report.html")
    with open(html_path, "w") as f:
        f.write(format_report_html(report))
    print(f"  Report (HTML) saved → {html_path}")
    paths["html"] = html_path

    # PDF
    paths["pdf"] = save_report_pdf(report, output_dir)

    return paths


# ═════════════════════════════════════════════════════════════
#  QUICK DEMO
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo_report = generate_clinical_report(
        pred_class   = 2,        # Wheeze
        confidence   = 87.3,
        all_probs    = {
            "Normal": 5.2, "Crackle": 4.1,
            "Wheeze": 87.3, "Both": 3.4
        },
        model_used   = "Ensemble (CNN + RF agreed)",
        time_taken   = 3.8,
        top_features = [
            {"feature": "ZCR",          "importance": 0.082},
            {"feature": "MFCC_mean_3",  "importance": 0.071},
            {"feature": "Delta_mean_2", "importance": 0.065},
            {"feature": "MFCC_std_5",   "importance": 0.058},
            {"feature": "RMS_energy",   "importance": 0.051},
        ],
        highlight_info = {"peak_time_sec": 2.4, "start_sec": 1.8, "end_sec": 3.1},
        patient_meta   = {"age": 58, "smoking": True, "copd_history": False}
    )

    print(format_report_text(demo_report))
    saved = save_report(demo_report)
    print("\nSaved files:", saved)