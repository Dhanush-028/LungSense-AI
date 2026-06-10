# ── standard library ──────────────────────────────────────────
import os
import json
import time
import warnings
import traceback

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import joblib

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS

# ── local ─────────────────────────────────────────────────────

from clinical_report import generate_clinical_report, format_report_text, save_report

# ═════════════════════════════════════════════════════════════
#  APP SETUP
# ═════════════════════════════════════════════════════════════
app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SAMPLE_RATE = 22050
IMG_SIZE    = 224
N_MELS      = 128
FMAX        = 8000

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp"}

# ── CORS / ngrok headers ──────────────────────────────────────
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin",  "*")
    response.headers.add("Access-Control-Allow-Headers",
                         "Content-Type,Authorization,ngrok-skip-browser-warning")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response


# ═════════════════════════════════════════════════════════════
#  LOAD MODELS AT STARTUP
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  LungSense AI — Loading Models")
print("=" * 55)

cnn_model    = None
idx_to_class = None

if os.path.exists("cnn_model.h5") and os.path.exists("cnn_class_indices.json"):
    import tensorflow as tf
    cnn_model = tf.keras.models.load_model("cnn_model.h5")
    with open("cnn_class_indices.json") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("  ✔ CNN MobileNetV2 loaded  (primary)")
else:
    print("  ✗ CNN model not found")

rf_model  = None
rf_le     = None
rf_scaler = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
for pkl in ["baseline_rf_model.pkl", "respiratory_model.pkl"]:
    pkl_path = os.path.join(BASE_DIR, pkl)
    if os.path.exists(pkl_path):
        d          = joblib.load(pkl_path)
        rf_model   = d["model"]
        rf_le      = d["label_encoder"]
        rf_scaler  = d.get("scaler", None)
        print(f"  ✓ Random Forest loaded from {pkl}  (fallback / ensemble)")
        break

if not rf_model:
    print("  ✗ Random Forest not found")

active_model = "ENSEMBLE" if (cnn_model and rf_model) else \
               "CNN"      if cnn_model else \
               "RF"       if rf_model  else None

print(f"  Active mode  : {active_model}")
print("=" * 55 + "\n")


# ═════════════════════════════════════════════════════════════
#  DISEASE INFO
# ═════════════════════════════════════════════════════════════
DISEASE_INFO = {
    "Normal": {
        "icon": "OK", "color": "#00ffa3", "urgency": "none",
        "advice":     "Your lung sounds appear normal. No abnormal patterns detected.",
        "conditions": "Healthy lungs",
        "pattern":    "Clean, clear airflow with no abnormal sounds."
    },
    "Crackle": {
        "icon": "WARN", "color": "#ffd166", "urgency": "moderate",
        "advice":     "Crackles detected. May indicate Pneumonia or Bronchitis. Please consult a doctor soon.",
        "conditions": "Pneumonia, Bronchitis, Heart failure",
        "pattern":    "Discontinuous crackling sounds during inhalation."
    },
    "Wheeze": {
        "icon": "HIGH", "color": "#ff8c42", "urgency": "high",
        "advice":     "Wheezing detected. May indicate Asthma or COPD. Please seek medical advice soon.",
        "conditions": "Asthma, COPD, Bronchospasm",
        "pattern":    "Continuous high-pitched whistling during breathing."
    },
    "Both": {
        "icon": "CRIT", "color": "#ff4d6d", "urgency": "critical",
        "advice":     "Both crackles and wheezes detected. Please see a doctor immediately.",
        "conditions": "Severe COPD, Mixed respiratory infection",
        "pattern":    "Both crackles and wheezes present simultaneously."
    },
    "Uncertain": {
        "icon": "?", "color": "#888780", "urgency": "review",
        "advice":     "Model confidence is too low for a reliable result. Re-record in a quieter environment or consult a doctor.",
        "conditions": "Inconclusive",
        "pattern":    "Audio pattern unclear or ambiguous."
    }
}

# class name → index for clinical_report
CLASS_TO_IDX = {"Normal": 0, "Crackle": 1, "Wheeze": 2, "Both": 3}


# ═════════════════════════════════════════════════════════════
#  AUDIO QUALITY CHECK
# ═════════════════════════════════════════════════════════════
def check_audio_quality(audio_path):
    """
    Returns (ok: bool, message: str).
    Rejects recordings that are:
      - too short  (< 2 s)
      - silent     (RMS < 0.001)
      - clipping   (RMS > 0.95)
      - too noisy  (spectral flatness > 0.6  →  mostly noise, not lung sound)
    """
    try:
        y, sr    = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=y, sr=sr)
        rms      = float(np.sqrt(np.mean(y ** 2)))

        if duration < 2.0:
            return False, f"Recording too short ({duration:.1f}s). Minimum 2 seconds needed."
        if rms < 0.001:
            return False, "Audio is too quiet. Check microphone placement on the chest."
        if rms > 0.95:
            return False, "Audio is clipping (too loud). Move microphone slightly away."

        # Spectral flatness — values near 1.0 = white noise, near 0 = tonal/structured
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        if flatness > 0.6:
            return False, "Too much background noise detected. Record in a quieter environment."

        return True, "ok"

    except Exception as e:
        return False, f"Could not read audio file: {str(e)}"


# ═════════════════════════════════════════════════════════════
#  INDIVIDUAL MODEL PIPELINES
# ═════════════════════════════════════════════════════════════

# ── CNN: WAV → mel-spectrogram image → MobileNetV2 ───────────
def predict_with_cnn(audio_path):
    from tensorflow.keras.preprocessing import image as keras_image

    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y, _  = librosa.effects.trim(y, top_db=20)

    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_db = librosa.power_to_db(S, ref=np.max)

    tmp_img = os.path.join(UPLOAD_FOLDER, "tmp_spec.png")
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_db, sr=sr, fmax=FMAX, ax=ax, cmap="magma")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(tmp_img, bbox_inches="tight", pad_inches=0)
    plt.close()

    img       = keras_image.load_img(tmp_img, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred_proba = cnn_model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(pred_proba))
    label      = idx_to_class[pred_idx]
    confidence = round(float(np.max(pred_proba)) * 100, 1)
    all_probs  = {idx_to_class[i]: round(float(p) * 100, 1)
                  for i, p in enumerate(pred_proba)}

    if os.path.exists(tmp_img):
        os.remove(tmp_img)

    return label, confidence, all_probs


# ── RF: WAV → MFCC features → Random Forest ─────────────────
def predict_with_rf(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y, _  = librosa.effects.trim(y)

    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40,
                                     n_fft=512, hop_length=256, n_mels=64)
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc,  axis=1),
        np.mean(librosa.feature.delta(mfcc), axis=1),
        [np.mean(librosa.feature.zero_crossing_rate(y))],
        [np.mean(librosa.feature.rms(y=y))]
    ]).reshape(1, -1)

    if rf_scaler:
        features = rf_scaler.transform(features)

    pred_enc   = rf_model.predict(features)[0]
    pred_proba = rf_model.predict_proba(features)[0]
    label      = rf_le.inverse_transform([pred_enc])[0]
    confidence = round(float(np.max(pred_proba)) * 100, 1)
    all_probs  = {c: round(float(p) * 100, 1)
                  for c, p in zip(rf_le.classes_, pred_proba)}

    return label, confidence, all_probs


# ═════════════════════════════════════════════════════════════
#  ENSEMBLE PREDICTION  (CNN + RF combined)
# ═════════════════════════════════════════════════════════════
def predict_ensemble(audio_path):
    """
    Runs both CNN and RF when available.

    Agreement  → average probabilities, small confidence boost (+5 %)
    Disagreement → pick the higher-confidence model, apply a
                   small penalty (-10 %) to signal uncertainty.

    Returns:
        label        : str   final predicted class
        confidence   : float final confidence %
        all_probs    : dict  {class: %}  averaged when possible
        model_note   : str   description of what happened
    """
    results = []

    if cnn_model:
        lbl, conf, probs = predict_with_cnn(audio_path)
        results.append({"model": "CNN", "label": lbl,
                         "confidence": conf, "probs": probs})

    if rf_model:
        lbl, conf, probs = predict_with_rf(audio_path)
        results.append({"model": "RF",  "label": lbl,
                         "confidence": conf, "probs": probs})

    # Only one model available — return as-is
    if len(results) == 1:
        r = results[0]
        return r["label"], r["confidence"], r["probs"], r["model"]

    cnn_r = results[0]
    rf_r  = results[1]

    # ── Merge probabilities (average) ────────────────────────
    all_classes = list(set(list(cnn_r["probs"].keys()) +
                            list(rf_r["probs"].keys())))
    merged_probs = {}
    for cls in all_classes:
        p_cnn = cnn_r["probs"].get(cls, 0.0)
        p_rf  = rf_r["probs"].get(cls, 0.0)
        merged_probs[cls] = round((p_cnn + p_rf) / 2, 1)

    # Normalise to 100 %
    total = sum(merged_probs.values())
    if total > 0:
        merged_probs = {k: round(v / total * 100, 1)
                        for k, v in merged_probs.items()}

    best_class = max(merged_probs, key=merged_probs.get)

    # ── Agreement vs disagreement ─────────────────────────────
    if cnn_r["label"] == rf_r["label"]:
        avg_conf   = round((cnn_r["confidence"] + rf_r["confidence"]) / 2, 1)
        # Small boost: both models agree
        final_conf = min(round(avg_conf + 5.0, 1), 99.0)
        note       = f"Ensemble — CNN & RF agreed ({cnn_r['confidence']}% / {rf_r['confidence']}%)"
    else:
        # Disagreement: pick higher-confidence model result, apply penalty
        best = max(results, key=lambda x: x["confidence"])
        best_class = best["label"]
        final_conf = round(best["confidence"] * 0.90, 1)   # 10 % penalty
        note       = (f"Ensemble — models disagreed "
                      f"(CNN:{cnn_r['label']} {cnn_r['confidence']}% | "
                      f"RF:{rf_r['label']} {rf_r['confidence']}%), "
                      f"used {best['model']}")

    return best_class, final_conf, merged_probs, note


# ═════════════════════════════════════════════════════════════
#  LOW-CONFIDENCE GUARD
# ═════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD = 30.0   # below this → "Uncertain"

def apply_confidence_guard(label, confidence, all_probs):
    """
    Overrides the predicted label to 'Uncertain' when the model
    is not confident enough. Returns updated (label, all_probs).
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain", all_probs
    return label, all_probs


# ═════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/diseases")
def diseases():
    return render_template("diseases.html")


@app.route('/games')
def games():
    return render_template('games.html')


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/history")
def history():
    records = get_all_history()
    return render_template("history.html", records=records)


@app.route("/history/delete/<int:record_id>", methods=["POST"])
def delete_history(record_id):
    delete_record(record_id)
    return redirect(url_for("history"))


# ── Main prediction endpoint ──────────────────────────────────
@app.route("/predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if request.method == "GET":
        return render_template("predict.html")

    # ── POST ─────────────────────────────────────────────────
    if not active_model:
        return jsonify({"error": "No model loaded. Run training scripts first."}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported format '{ext}'. "
                                  f"Accepted: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    try:
        # ── Step 1: audio quality check ──────────────────────
        quality_ok, quality_msg = check_audio_quality(save_path)
        if not quality_ok:
            os.remove(save_path)
            return jsonify({
                "error":        quality_msg,
                "quality_fail": True
            }), 422   # Unprocessable Entity

        # ── Step 2: run prediction ────────────────────────────
        start = time.time()

        if active_model == "ENSEMBLE":
            label, confidence, all_probs, model_used = predict_ensemble(save_path)
        elif active_model == "CNN":
            label, confidence, all_probs = predict_with_cnn(save_path)
            model_used = "CNN MobileNetV2"
        else:
            label, confidence, all_probs = predict_with_rf(save_path)
            model_used = "Random Forest"

        elapsed = round(time.time() - start, 2)

        # ── Step 3: low-confidence guard ─────────────────────
        label, all_probs = apply_confidence_guard(label, confidence, all_probs)

        # ── Step 4: build response ────────────────────────────
        info = DISEASE_INFO.get(label, DISEASE_INFO["Uncertain"])

        # ── Step 5: generate clinical report ─────────────────
        pred_idx = CLASS_TO_IDX.get(label, 0)
        report   = generate_clinical_report(
            pred_class  = pred_idx,
            confidence  = confidence,
            all_probs   = all_probs,
            model_used  = model_used,
            time_taken  = elapsed
        )

        # ── Step 6: save to database ──────────────────────────
        try:
            save_prediction(
                filename   = file.filename,
                disease    = label,
                confidence = confidence,
                model_used = model_used
            )
        except Exception as db_err:
            print(f"  DB save warning: {db_err}")

        # ── Step 7: cleanup + respond ─────────────────────────
        if os.path.exists(save_path):
            os.remove(save_path)

        return jsonify({
            # ── Core prediction ───────────────────────────────
            "disease":         label,
            "confidence":      confidence,
            "confident_enough": confidence >= CONFIDENCE_THRESHOLD,

            # ── Clinical info ─────────────────────────────────
            "message":         info["advice"],
            "color":           info["color"],
            "urgency":         info["urgency"],
            "conditions":      info["conditions"],
            "pattern":         info["pattern"],

            # ── Probabilities ─────────────────────────────────
            "all_probs":       all_probs,

            # ── Meta ──────────────────────────────────────────
            "model_used":      model_used,
            "time_taken":      elapsed,

            # ── Risk from clinical report ─────────────────────
            "risk_score":      report["risk"]["score"],
            "risk_label":      report["risk"]["label"],
            "risk_action":     report["risk"]["action"],
            "risk_color":      report["risk"]["color"],

            # ── Doctor note ───────────────────────────────────
            "doctor_note":     report["doctor_note"],
            "next_steps":      report["advice"]["next_steps"],
        })

    except Exception as e:
        traceback.print_exc()
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ═════════════════════════════════════════════════════════════
#  API STATUS
# ═════════════════════════════════════════════════════════════
@app.route("/api/status")
def status():
    return jsonify({
        "active_mode":      active_model,
        "cnn_loaded":       cnn_model  is not None,
        "rf_loaded":        rf_model   is not None,
        "ensemble_enabled": active_model == "ENSEMBLE",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })


# ═════════════════════════════════════════════════════════════
#  RUN
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Open browser : http://127.0.0.1:5000")
    print("  For Android  : use ngrok URL")
    print("=" * 55 + "\n")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)