import os
import librosa
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATA_FOLDER  = "audio_and_txt_files"
MODEL_PATH   = "baseline_rf_model.pkl"
SAMPLE_RATE  = 22050
N_MFCC       = 40
N_FFT        = 512
HOP_LENGTH   = 256
N_MELS       = 64
MIN_SAMPLES  = 2000

# ─────────────────────────────────────────
# LABEL FUNCTION
# ─────────────────────────────────────────
def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0: return "Normal"
    elif crackle == 1 and wheeze == 0: return "Crackle"
    elif crackle == 0 and wheeze == 1: return "Wheeze"
    else: return "Both"

# ─────────────────────────────────────────
# FEATURE EXTRACTION  (122 features/segment)
# ─────────────────────────────────────────
def extract_features(segment, sr):
    # MFCC — 40 coefficients (mean + std = 80)
    mfcc       = librosa.feature.mfcc(
                     y=segment, sr=sr, n_mfcc=N_MFCC,
                     n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    mfcc_mean  = np.mean(mfcc, axis=1)   # 40
    mfcc_std   = np.std(mfcc,  axis=1)   # 40

    # Delta MFCC — how features change over time (40)
    delta      = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)  # 40

    # Zero Crossing Rate — distinguishes voiced/unvoiced (1)
    zcr        = np.mean(librosa.feature.zero_crossing_rate(segment))

    # RMS Energy — loudness of segment (1)
    rms        = np.mean(librosa.feature.rms(y=segment))

    # Total = 40 + 40 + 40 + 1 + 1 = 122 features
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, [zcr], [rms]])

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
print("=" * 55)
print("  Baseline Model — Random Forest (MFCC + ZCR + RMS)")
print("=" * 55)
print("\n[1/4] Extracting features from audio files...")

all_features = []
all_labels   = []
skipped      = 0
processed    = 0

wav_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".wav")]
total     = len(wav_files)

for idx, file in enumerate(wav_files):
    audio_path = os.path.join(DATA_FOLDER, file)
    txt_path   = audio_path.replace(".wav", ".txt")

    if not os.path.exists(txt_path):
        skipped += 1
        continue

    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        y, _  = librosa.effects.trim(y, top_db=20)

        df = pd.read_csv(txt_path, sep='\t', header=None)
        df.columns = ["start", "end", "crackle", "wheeze"]

        for _, row in df.iterrows():
            start   = int(row["start"] * sr)
            end     = int(row["end"]   * sr)
            segment = y[start:end]

            if len(segment) < MIN_SAMPLES:
                continue

            features = extract_features(segment, sr)
            label    = get_label(int(row["crackle"]), int(row["wheeze"]))

            all_features.append(features)
            all_labels.append(label)
            processed += 1

    except Exception as e:
        print(f"  ⚠ Skipped {file}: {e}")
        skipped += 1

    if (idx + 1) % 50 == 0:
        print(f"  [{idx+1}/{total}] files done...")

X = np.array(all_features)
y = np.array(all_labels)

print(f"\n  ✔ Segments extracted : {processed}")
print(f"  ✔ Feature shape      : {X.shape}  ← (segments × 122 features)")
print(f"  ✔ Files skipped      : {skipped}")

# ─────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\n[2/4] Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    pct = cnt / len(y) * 100
    bar = "█" * (cnt // 50)
    print(f"  {cls:<10} {cnt:>5} ({pct:.1f}%)  {bar}")

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler

le = LabelEncoder()
y_enc = le.fit_transform(y)

# StandardScaler — normalise features for better RF performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_enc       # keeps class ratio in both splits
)

print(f"\n  Train segments : {len(X_train)}")
print(f"  Test  segments : {len(X_test)}")

# ─────────────────────────────────────────
# TRAIN RANDOM FOREST
# ─────────────────────────────────────────
print("\n[3/4] Training Random Forest (class_weight='balanced')...")

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,          # 200 trees
    max_depth=None,            # fully grown trees
    min_samples_split=5,       # slightly reduces overfitting
    class_weight="balanced",   # handles class imbalance automatically
    random_state=42,
    n_jobs=-1                  # use all CPU cores
)
rf_model.fit(X_train, y_train)
print("  ✔ Training complete!")

# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
print("\n[4/4] Evaluation Results:")
print("-" * 55)

from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              accuracy_score,
                              f1_score)

y_pred = rf_model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average='weighted')

print(f"\n  Overall Accuracy     : {acc * 100:.2f}%")
print(f"  Weighted F1-Score    : {f1  * 100:.2f}%")
print(f"\n  Per-class Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    digits=3
))

# ── Confusion Matrix ────────────────────
print("  Confusion Matrix (rows=Actual, cols=Predicted):\n")
cm     = confusion_matrix(y_test, y_pred)
labels = le.classes_
header = "  " + " " * 10 + "".join(f"{l:>10}" for l in labels)
print(header)
for i, row in enumerate(cm):
    row_str = f"  {labels[i]:<10}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# ── Feature Importance (top 10) ─────────
print("\n  Top 10 most important features:")
feat_names = (
    [f"MFCC_mean_{i}"  for i in range(40)] +
    [f"MFCC_std_{i}"   for i in range(40)] +
    [f"Delta_mean_{i}" for i in range(40)] +
    ["ZCR", "RMS"]
)
importances = rf_model.feature_importances_
top10_idx   = np.argsort(importances)[::-1][:10]
for rank, i in enumerate(top10_idx, 1):
    print(f"  {rank:>2}. {feat_names[i]:<20} {importances[i]:.4f}")

# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────
joblib.dump({
    "model":         rf_model,
    "label_encoder": le,
    "scaler":        scaler
}, MODEL_PATH)

print(f"\n  ✔ Model saved → {MODEL_PATH}")
print("\n" + "=" * 55)
print(f"  Baseline accuracy: {acc * 100:.2f}%")
print("  Next: run train_cnn.py to compare CNN vs this")
print("=" * 55)