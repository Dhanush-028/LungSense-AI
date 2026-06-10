import librosa
import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
DATA_FOLDER   = "audio_and_txt_files"
MODEL_PATH    = "respiratory_model.pkl"
N_MFCC        = 40        # upgraded from 13 → richer features
N_FFT         = 512
HOP_LENGTH    = 256
N_MELS        = 64
MIN_SAMPLES   = 2000

# ─────────────────────────────────────────
# LABEL FUNCTION
# ─────────────────────────────────────────
def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0:
        return "Normal"
    elif crackle == 1 and wheeze == 0:
        return "Crackle"
    elif crackle == 0 and wheeze == 1:
        return "Wheeze"
    else:
        return "Both"

# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(segment, sr):
    """Extract MFCC + Delta + Delta-Delta + ZCR + RMS features."""
    # MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(
        y=segment, sr=sr,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mfcc_mean  = np.mean(mfcc, axis=1)       # shape: (40,)
    mfcc_std   = np.std(mfcc, axis=1)        # shape: (40,)

    # Delta MFCC — captures how sound changes over time
    delta      = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)      # shape: (40,)

    # Zero Crossing Rate — distinguishes voiced/unvoiced sounds
    zcr        = librosa.feature.zero_crossing_rate(segment)
    zcr_mean   = np.mean(zcr)               # shape: (1,)

    # RMS Energy — volume/loudness of segment
    rms        = librosa.feature.rms(y=segment)
    rms_mean   = np.mean(rms)               # shape: (1,)

    # Combine all → 122 total features
    features = np.concatenate([
        mfcc_mean, mfcc_std, delta_mean,
        [zcr_mean], [rms_mean]
    ])
    return features

# ─────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────
print("=" * 50)
print("  Respiratory Disease Classifier — ICBHI 2017")
print("=" * 50)
print("\n[1/4] Loading and extracting features...")

all_features = []
all_labels   = []
skipped      = 0
processed    = 0

if os.path.exists(DATA_FOLDER):
    wav_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".wav")]
else:
    wav_files = []
total = len(wav_files)

for idx, file in enumerate(wav_files):
    audio_path = os.path.join(DATA_FOLDER, file)
    txt_path   = audio_path.replace(".wav", ".txt")

    if not os.path.exists(txt_path):
        skipped += 1
        continue

    try:
        y_audio, sr = librosa.load(audio_path, sr=22050)  # resample to 22050 Hz

        df = pd.read_csv(txt_path, sep='\t', header=None)
        df.columns = ["start", "end", "crackle", "wheeze"]
        df["label"] = df.apply(
            lambda x: get_label(int(x["crackle"]), int(x["wheeze"])), axis=1
        )

        for _, row in df.iterrows():
            start   = int(row["start"] * sr)
            end     = int(row["end"]   * sr)
            segment = y_audio[start:end]

            if len(segment) < MIN_SAMPLES:
                continue

            features = extract_features(segment, sr)
            all_features.append(features)
            all_labels.append(row["label"])
            processed += 1

    except Exception as e:
        print(f"  ⚠ Skipped {file}: {e}")
        skipped += 1

    # Progress indicator every 50 files
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{total} files...")

print(f"\n  ✔ Segments extracted : {processed}")
print(f"  ✔ Files skipped      : {skipped}")

X = np.array(all_features)
y = np.array(all_labels)
print(f"  ✔ Feature matrix     : {X.shape}")

# ─────────────────────────────────────────
# CLASS DISTRIBUTION
# ─────────────────────────────────────────
print("\n[2/4] Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, cnt in zip(unique, counts):
    bar = "█" * (cnt // 50)
    print(f"  {cls:<10} {cnt:>5}  {bar}")

# ─────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded      # ensures balanced split across all classes
)

print(f"\n  Train samples : {len(X_train)}")
print(f"  Test  samples : {len(X_test)}")

# ─────────────────────────────────────────
# TRAIN MODEL (with class balancing)
# ─────────────────────────────────────────
print("\n[3/4] Training Random Forest model...")

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,         # upgraded from 100 → more trees = better accuracy
    max_depth=None,           # let trees grow fully
    class_weight="balanced",  # 🔑 fixes class imbalance automatically
    random_state=42,
    n_jobs=-1                 # use all CPU cores for speed
)
model.fit(X_train, y_train)
print("  ✔ Training complete!")

# ─────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────
print("\n[4/4] Evaluation Results:")
print("-" * 50)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n  Overall Accuracy: {acc * 100:.2f}%\n")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# Confusion Matrix
print("  Confusion Matrix:")
print("  (rows = Actual, cols = Predicted)\n")
cm     = confusion_matrix(y_test, y_pred)
labels = le.classes_
header = "  " + "".join(f"{l:>10}" for l in labels)
print(header)
for i, row in enumerate(cm):
    row_str = "  " + f"{labels[i]:<10}" + "".join(f"{v:>10}" for v in row)
    print(row_str)

# ─────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────
joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)
print(f"\n  ✔ Model saved → {MODEL_PATH}")
print("\n" + "=" * 50)
print("  Done! Your model is ready.")
print("=" * 50)