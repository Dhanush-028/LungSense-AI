"""
compare_models.py  —  LungSense AI  (updated)
==============================================
Improvements:
  1. ROC curve  (macro-average AUC)
  2. Per-class sensitivity & specificity
  3. 3-panel figure: bar chart + confusion matrix + ROC curve
  4. Saves a summary JSON for the web app to display
"""

import os
import numpy as np
import joblib
import json
import librosa
import warnings
warnings.filterwarnings("ignore")

print("=" * 65)
print("  Model Comparison — CNN MobileNetV2  vs  Random Forest")
print("=" * 65)

# ═════════════════════════════════════════════════════════════
#  CHECK FILES EXIST
# ═════════════════════════════════════════════════════════════
missing = []
if not os.path.exists("baseline_rf_model.pkl"):
    missing.append("baseline_rf_model.pkl  ← run baseline_model.py")
if not os.path.exists("cnn_model.h5"):
    missing.append("cnn_model.h5           ← run train_cnn.py")
if not os.path.exists("spectrograms"):
    missing.append("spectrograms/          ← run preprocess.py")
if not os.path.exists("audio_and_txt_files"):
    missing.append("audio_and_txt_files/   ← ICBHI dataset folder")

if missing:
    print("\n❌ Missing files — please run these first:")
    for m in missing:
        print(f"   {m}")
    exit(1)


# ═════════════════════════════════════════════════════════════
#  LOAD BOTH MODELS
# ═════════════════════════════════════════════════════════════
print("\n[1/5] Loading models...")

rf_data   = joblib.load("baseline_rf_model.pkl")
rf_model  = rf_data["model"]
rf_le     = rf_data["label_encoder"]
rf_scaler = rf_data["scaler"]
print("  ✔ Random Forest loaded")

import tensorflow as tf
cnn_model = tf.keras.models.load_model("cnn_model.h5")
with open("cnn_class_indices.json") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
cnn_labels   = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
print("  ✔ CNN MobileNetV2 loaded")


# ═════════════════════════════════════════════════════════════
#  PREPARE TEST DATA
# ═════════════════════════════════════════════════════════════
print("\n[2/5] Preparing test data...")

DATA_FOLDER = "audio_and_txt_files"
SAMPLE_RATE = 22050
IMG_SIZE    = 224

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder

def get_label(c, w):
    if   c == 0 and w == 0: return "Normal"
    elif c == 1 and w == 0: return "Crackle"
    elif c == 0 and w == 1: return "Wheeze"
    else:                    return "Both"

def extract_rf_features(segment, sr):
    mfcc       = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40,
                                       n_fft=512, hop_length=256, n_mels=64)
    delta      = librosa.feature.delta(mfcc)
    return np.concatenate([
        np.mean(mfcc,  axis=1),
        np.std(mfcc,   axis=1),
        np.mean(delta, axis=1),
        [np.mean(librosa.feature.zero_crossing_rate(segment))],
        [np.mean(librosa.feature.rms(y=segment))]
    ])

all_features, all_labels = [], []

for file in os.listdir(DATA_FOLDER):
    if not file.endswith(".wav"):
        continue
    audio_path = os.path.join(DATA_FOLDER, file)
    txt_path   = audio_path.replace(".wav", ".txt")
    if not os.path.exists(txt_path):
        continue
    try:
        y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        y_audio, _  = librosa.effects.trim(y_audio)
        df = pd.read_csv(txt_path, sep="\t", header=None,
                         names=["start", "end", "crackle", "wheeze"])
        for _, row in df.iterrows():
            seg = y_audio[int(row["start"] * sr): int(row["end"] * sr)]
            if len(seg) < 2000:
                continue
            all_features.append(extract_rf_features(seg, sr))
            all_labels.append(get_label(int(row["crackle"]),
                                         int(row["wheeze"])))
    except Exception:
        continue

X     = np.array(all_features)
y     = np.array(all_labels)
le    = LabelEncoder()
y_enc = le.fit_transform(y)
X_sc  = rf_scaler.transform(X)

_, X_test_rf, _, y_test_rf = train_test_split(
    X_sc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# CNN validation generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
val_gen     = val_datagen.flow_from_directory(
    "spectrograms",
    target_size  = (IMG_SIZE, IMG_SIZE),
    batch_size   = 32,
    class_mode   = "categorical",
    subset       = "validation",
    shuffle      = False,
    seed         = 42
)


# ═════════════════════════════════════════════════════════════
#  PREDICTIONS
# ═════════════════════════════════════════════════════════════
print("\n[3/5] Running predictions...")

# Random Forest
rf_pred       = rf_model.predict(X_test_rf)
rf_pred_proba = rf_model.predict_proba(X_test_rf)
rf_true       = y_test_rf

# CNN
val_gen.reset()
cnn_pred_proba = cnn_model.predict(val_gen, verbose=0)
cnn_pred       = np.argmax(cnn_pred_proba, axis=1)
cnn_true       = val_gen.classes


# ═════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════
print("\n[4/5] Computing metrics...")

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

def sensitivity_specificity(y_true, y_pred, classes):
    """Returns per-class sensitivity (recall) and specificity."""
    cm = confusion_matrix(y_true, y_pred)
    results = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results[cls] = {
            "sensitivity": round(sensitivity * 100, 2),
            "specificity": round(specificity * 100, 2)
        }
    return results

# Scalar metrics
rf_acc   = accuracy_score(rf_true, rf_pred)       * 100
cnn_acc  = accuracy_score(cnn_true, cnn_pred)     * 100
rf_f1    = f1_score(rf_true,  rf_pred,  average="weighted") * 100
cnn_f1   = f1_score(cnn_true, cnn_pred, average="weighted") * 100
rf_prec  = precision_score(rf_true,  rf_pred,  average="weighted") * 100
cnn_prec = precision_score(cnn_true, cnn_pred, average="weighted") * 100
rf_rec   = recall_score(rf_true,  rf_pred,  average="weighted") * 100
cnn_rec  = recall_score(cnn_true, cnn_pred, average="weighted") * 100

# Confusion matrices
cm_rf  = confusion_matrix(rf_true, rf_pred)
cm_cnn = confusion_matrix(cnn_true, cnn_pred)

# Per-class sensitivity / specificity
rf_ss  = sensitivity_specificity(rf_true,  rf_pred,  list(range(len(le.classes_))))
cnn_ss = sensitivity_specificity(cnn_true, cnn_pred, list(range(len(cnn_labels))))

# ROC AUC  (macro-average, one-vs-rest)
n_rf_classes  = len(le.classes_)
n_cnn_classes = len(cnn_labels)

rf_true_bin  = label_binarize(rf_true,  classes=list(range(n_rf_classes)))
cnn_true_bin = label_binarize(cnn_true, classes=list(range(n_cnn_classes)))

rf_fpr,  rf_tpr,  _ = roc_curve(rf_true_bin.ravel(),  rf_pred_proba.ravel())
cnn_fpr, cnn_tpr, _ = roc_curve(cnn_true_bin.ravel(), cnn_pred_proba.ravel())
rf_auc  = auc(rf_fpr,  rf_tpr)
cnn_auc = auc(cnn_fpr, cnn_tpr)

winner_acc  = "CNN" if cnn_acc  > rf_acc  else "RF"
winner_f1   = "CNN" if cnn_f1   > rf_f1   else "RF"
winner_prec = "CNN" if cnn_prec > rf_prec else "RF"
winner_rec  = "CNN" if cnn_rec  > rf_rec  else "RF"
winner_auc  = "CNN" if cnn_auc  > rf_auc  else "RF"


# ═════════════════════════════════════════════════════════════
#  PRINT RESULTS
# ═════════════════════════════════════════════════════════════
print("\n[5/5] Results:")
print("=" * 65)
print(f"\n  {'Metric':<24} {'Random Forest':>14} {'CNN':>10} {'Winner':>8}")
print("  " + "-" * 58)
print(f"  {'Accuracy':<24} {rf_acc:>13.2f}% {cnn_acc:>9.2f}% {winner_acc:>8}")
print(f"  {'Weighted F1':<24} {rf_f1:>13.2f}% {cnn_f1:>9.2f}% {winner_f1:>8}")
print(f"  {'Precision':<24} {rf_prec:>13.2f}% {cnn_prec:>9.2f}% {winner_prec:>8}")
print(f"  {'Recall':<24} {rf_rec:>13.2f}% {cnn_rec:>9.2f}% {winner_rec:>8}")
print(f"  {'ROC AUC (macro)':<24} {rf_auc:>13.3f}  {cnn_auc:>9.3f}  {winner_auc:>8}")
print("  " + "-" * 58)

best_model = "CNN MobileNetV2" if cnn_acc > rf_acc else "Random Forest"
diff = abs(cnn_acc - rf_acc)
print(f"\n  🏆 Best model: {best_model}  (+{diff:.2f}% accuracy)")

# Per-class breakdown
print("\n" + "=" * 65)
print("  Per-class Report — Random Forest:")
print("-" * 65)
print(classification_report(rf_true, rf_pred,
                              target_names=le.classes_, digits=3))

print("  Per-class Report — CNN MobileNetV2:")
print("-" * 65)
print(classification_report(cnn_true, cnn_pred,
                              target_names=cnn_labels, digits=3))

# Sensitivity / Specificity
print("  Sensitivity & Specificity — Random Forest:")
print(f"  {'Class':<12} {'Sensitivity':>14} {'Specificity':>14}")
print("  " + "-" * 42)
for idx, cls in enumerate(le.classes_):
    ss = rf_ss[idx]
    print(f"  {cls:<12} {ss['sensitivity']:>13.2f}% {ss['specificity']:>13.2f}%")

print("\n  Sensitivity & Specificity — CNN MobileNetV2:")
print(f"  {'Class':<12} {'Sensitivity':>14} {'Specificity':>14}")
print("  " + "-" * 42)
for idx, cls in enumerate(cnn_labels):
    ss = cnn_ss[idx]
    print(f"  {cls:<12} {ss['sensitivity']:>13.2f}% {ss['specificity']:>13.2f}%")

# Confusion matrices
print("\n  Confusion Matrix — Random Forest:")
header = "  " + " " * 10 + "".join(f"{l:>10}" for l in le.classes_)
print(header)
for i, row in enumerate(cm_rf):
    print(f"  {le.classes_[i]:<10}" + "".join(f"{v:>10}" for v in row))

print("\n  Confusion Matrix — CNN MobileNetV2:")
header = "  " + " " * 10 + "".join(f"{l:>10}" for l in cnn_labels)
print(header)
for i, row in enumerate(cm_cnn):
    print(f"  {cnn_labels[i]:<10}" + "".join(f"{v:>10}" for v in row))


# ═════════════════════════════════════════════════════════════
#  SAVE COMPARISON PLOT  (3 panels)
# ═════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG_DARK  = "#0a1628"
BG_PANEL = "#0f1e35"
SPINE    = "#1e3a5f"
WHITE    = "white"
BLUE     = "#0072ff"
GREEN    = "#00ffa3"
AMBER    = "#ffd166"

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor(BG_DARK)

# ── Panel 1: Bar chart ────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor(BG_PANEL)
metrics    = ["Accuracy", "F1", "Precision", "Recall"]
rf_scores  = [rf_acc,  rf_f1,  rf_prec,  rf_rec]
cnn_scores = [cnn_acc, cnn_f1, cnn_prec, cnn_rec]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax1.bar(x - w / 2, rf_scores,  w, label="Random Forest",   color=BLUE,  alpha=0.85)
bars2 = ax1.bar(x + w / 2, cnn_scores, w, label="CNN MobileNetV2", color=GREEN, alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, color=WHITE)
ax1.set_ylim(0, 115)
ax1.set_ylabel("Score (%)", color=WHITE)
ax1.set_title("CNN vs Random Forest — All Metrics",
              color=WHITE, fontsize=12, fontweight="bold")
ax1.tick_params(colors=WHITE)
for sp in ax1.spines.values(): sp.set_color(SPINE)
ax1.legend(facecolor=BG_PANEL, labelcolor=WHITE)
for bar in (*bars1, *bars2):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f"{bar.get_height():.1f}%",
             ha="center", va="bottom", color=WHITE, fontsize=8)

# ── Panel 2: CNN confusion matrix ─────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG_PANEL)
im = ax2.imshow(cm_cnn, cmap="Blues")
ax2.set_xticks(range(len(cnn_labels)))
ax2.set_yticks(range(len(cnn_labels)))
ax2.set_xticklabels(cnn_labels, rotation=45, ha="right", color=WHITE)
ax2.set_yticklabels(cnn_labels, color=WHITE)
ax2.set_title("CNN Confusion Matrix",
              color=WHITE, fontsize=12, fontweight="bold")
ax2.set_xlabel("Predicted", color=WHITE)
ax2.set_ylabel("Actual",    color=WHITE)
ax2.tick_params(colors=WHITE)
for sp in ax2.spines.values(): sp.set_color(SPINE)
thresh = cm_cnn.max() / 2
for i in range(len(cnn_labels)):
    for j in range(len(cnn_labels)):
        ax2.text(j, i, str(cm_cnn[i, j]),
                 ha="center", va="center",
                 color="black" if cm_cnn[i, j] > thresh else WHITE,
                 fontweight="bold")

# ── Panel 3: ROC curve ────────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor(BG_PANEL)
ax3.plot(cnn_fpr, cnn_tpr, color=GREEN,  lw=2,
         label=f"CNN  AUC = {cnn_auc:.3f}")
ax3.plot(rf_fpr,  rf_tpr,  color=BLUE,   lw=2,
         label=f"RF   AUC = {rf_auc:.3f}")
ax3.plot([0, 1], [0, 1],   color="gray", lw=1, linestyle="--",
         label="Random baseline")
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel("False Positive Rate", color=WHITE)
ax3.set_ylabel("True Positive Rate",  color=WHITE)
ax3.set_title("ROC Curve (macro-average)",
              color=WHITE, fontsize=12, fontweight="bold")
ax3.tick_params(colors=WHITE)
for sp in ax3.spines.values(): sp.set_color(SPINE)
ax3.legend(facecolor=BG_PANEL, labelcolor=WHITE)

plt.tight_layout()
plt.savefig("comparison_results.png",
            dpi=150, bbox_inches="tight", facecolor=BG_DARK)
plt.close()
print("\n  ✔ Comparison chart saved → comparison_results.png")


# ═════════════════════════════════════════════════════════════
#  SAVE SUMMARY JSON  (for web app /api/model_summary)
# ═════════════════════════════════════════════════════════════
summary = {
    "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
    "random_forest": {
        "accuracy":  round(rf_acc,  2),
        "f1":        round(rf_f1,   2),
        "precision": round(rf_prec, 2),
        "recall":    round(rf_rec,  2),
        "roc_auc":   round(rf_auc,  4),
    },
    "cnn": {
        "accuracy":  round(cnn_acc,  2),
        "f1":        round(cnn_f1,   2),
        "precision": round(cnn_prec, 2),
        "recall":    round(cnn_rec,  2),
        "roc_auc":   round(cnn_auc,  4),
    },
    "best_model":            best_model,
    "accuracy_improvement":  round(diff, 2),
    "per_class": {
        "cnn": {
            cls: cnn_ss[idx]
            for idx, cls in enumerate(cnn_labels)
        },
        "rf": {
            cls: rf_ss[idx]
            for idx, cls in enumerate(le.classes_)
        }
    }
}

with open("model_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  ✔ Summary JSON saved → model_summary.json")


# ═════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  Summary for project report:")
print(f"  Random Forest  accuracy : {rf_acc:.2f}%   AUC: {rf_auc:.3f}")
print(f"  CNN MobileNetV2 accuracy: {cnn_acc:.2f}%   AUC: {cnn_auc:.3f}")
print(f"  Best model              : {best_model}")
print("=" * 65)