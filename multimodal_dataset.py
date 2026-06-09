"""
multimodal_dataset.py
=====================
Loads ICBHI 2017 dataset as multi-modal samples.
Each sample = (MFCC features, spectrogram image, patient metadata, label)
"""

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.preprocessing import image as keras_image

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
DATA_FOLDER  = "audio_and_txt_files"
DEMO_CSV     = "demographic_info.csv"   # patient age, sex, BMI etc.
DIAG_CSV     = "patient_diagnosis.csv"  # patient disease label
SPEC_FOLDER  = "spectrograms"
SAMPLE_RATE  = 22050
N_MFCC       = 40
IMG_SIZE     = 224
MIN_SAMPLES  = 2000

LABEL_MAP = {
    "Normal":  0,
    "Crackle": 1,
    "Wheeze":  2,
    "Both":    3
}

# ─────────────────────────────────────────
# LABEL FUNCTION
# ─────────────────────────────────────────
def get_label(crackle, wheeze):
    if crackle == 0 and wheeze == 0: return "Normal"
    elif crackle == 1 and wheeze == 0: return "Crackle"
    elif crackle == 0 and wheeze == 1: return "Wheeze"
    else: return "Both"

# ─────────────────────────────────────────
# MFCC FEATURE EXTRACTION (122 features)
# ─────────────────────────────────────────
def extract_mfcc(segment, sr):
    mfcc       = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC,
                                       n_fft=512, hop_length=256, n_mels=64)
    mfcc_mean  = np.mean(mfcc, axis=1)   # 40
    mfcc_std   = np.std(mfcc,  axis=1)   # 40
    delta_mean = np.mean(librosa.feature.delta(mfcc), axis=1)  # 40
    zcr        = np.mean(librosa.feature.zero_crossing_rate(segment))  # 1
    rms        = np.mean(librosa.feature.rms(y=segment))               # 1
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean, [zcr], [rms]])
    # Total: 122 features

# ─────────────────────────────────────────
# SPECTROGRAM → IMAGE
# ─────────────────────────────────────────
def segment_to_image(segment, sr, tmp_path="tmp_spec.png"):
    S    = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    librosa.display.specshow(S_db, sr=sr, fmax=8000, ax=ax, cmap='magma')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(tmp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = keras_image.load_img(tmp_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img) / 255.0
    os.remove(tmp_path)
    return arr   # shape: (224, 224, 3)

# ─────────────────────────────────────────
# LOAD PATIENT METADATA
# ─────────────────────────────────────────
def load_metadata():
    """
    Returns dict: {patient_id: metadata_vector (8,)}
    Metadata vector: [age_norm, bmi_norm, is_male, is_smoker,
                      has_cough, has_wheeze, has_chest_pain, has_dyspnea]
    If demographic_info.csv not found, returns empty dict (metadata will be zeros).
    """
    meta = {}

    if not os.path.exists(DEMO_CSV):
        print(f"  ⚠ {DEMO_CSV} not found — metadata will be zeros")
        return meta

    df = pd.read_csv(DEMO_CSV)
    # Expected columns: Patient number, Age, Sex, BMI, Weight, Height, ...
    for _, row in df.iterrows():
        pid = int(row.get("Patient number", row.iloc[0]))
        age  = float(row.get("Age", 40)) / 100.0   # normalize 0–1
        bmi  = float(row.get("BMI", 22)) / 50.0
        sex  = 1.0 if str(row.get("Sex","M")).upper() == "M" else 0.0
        meta[pid] = np.array([age, bmi, sex, 0, 0, 0, 0, 0], dtype=np.float32)
    return meta

# ─────────────────────────────────────────
# MAIN DATASET LOADER
# ─────────────────────────────────────────
def load_multimodal_dataset(max_samples=None, save_images=False):
    """
    Returns:
        X_mfcc   : np.array (N, 122)
        X_images : np.array (N, 224, 224, 3)
        X_meta   : np.array (N, 8)
        y        : np.array (N,)  — integer labels 0–3
        labels   : list of string labels
    """
    print("="*58)
    print("  Loading multi-modal ICBHI dataset")
    print("="*58)

    meta_dict = load_metadata()

    all_mfcc, all_images, all_meta, all_labels = [], [], [], []
    processed = 0
    skipped   = 0

    wav_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".wav")])
    total     = len(wav_files)
    print(f"\n  Found {total} WAV files\n")

    for idx, file in enumerate(wav_files):
        if max_samples and processed >= max_samples:
            break

        audio_path = os.path.join(DATA_FOLDER, file)
        txt_path   = audio_path.replace(".wav", ".txt")
        if not os.path.exists(txt_path):
            skipped += 1
            continue

        # Extract patient ID from filename
        try:
            patient_id = int(file.split("_")[0])
        except:
            patient_id = -1

        try:
            y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            y_audio, _  = librosa.effects.trim(y_audio, top_db=20)

            df_ann = pd.read_csv(txt_path, sep='\t', header=None)
            df_ann.columns = ["start", "end", "crackle", "wheeze"]

            for _, row in df_ann.iterrows():
                start   = int(row["start"] * sr)
                end     = int(row["end"]   * sr)
                segment = y_audio[start:end]

                if len(segment) < MIN_SAMPLES:
                    continue

                label_str = get_label(int(row["crackle"]), int(row["wheeze"]))
                label_int = LABEL_MAP[label_str]

                # 1. MFCC features (122,)
                mfcc_feat = extract_mfcc(segment, sr)

                # 2. Spectrogram image (224, 224, 3)
                tmp = f"tmp_{idx}_{processed}.png"
                img_arr = segment_to_image(segment, sr, tmp)

                # 3. Patient metadata (8,)
                meta_vec = meta_dict.get(
                    patient_id,
                    np.zeros(8, dtype=np.float32)
                )

                all_mfcc.append(mfcc_feat)
                all_images.append(img_arr)
                all_meta.append(meta_vec)
                all_labels.append(label_int)
                processed += 1

        except Exception as e:
            print(f"  ⚠ Skipped {file}: {e}")
            skipped += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{total}] files, {processed} segments...")

    X_mfcc   = np.array(all_mfcc,   dtype=np.float32)
    X_images = np.array(all_images, dtype=np.float32)
    X_meta   = np.array(all_meta,   dtype=np.float32)
    y        = np.array(all_labels, dtype=np.int32)

    print(f"\n  Segments loaded : {processed}")
    print(f"  Files skipped   : {skipped}")
    print(f"  MFCC shape      : {X_mfcc.shape}")
    print(f"  Image shape     : {X_images.shape}")
    print(f"  Meta shape      : {X_meta.shape}")
    print(f"  Labels shape    : {y.shape}")

    # Class distribution
    print("\n  Class distribution:")
    for name, idx_l in LABEL_MAP.items():
        cnt = np.sum(y == idx_l)
        bar = "█" * (cnt // 50)
        print(f"    {name:<10} {cnt:>5}  {bar}")

    print("="*58)
    return X_mfcc, X_images, X_meta, y


# ─────────────────────────────────────────
# ONE-HOT ENCODE LABELS
# ─────────────────────────────────────────
def to_categorical(y, num_classes=4):
    return np.eye(num_classes, dtype=np.float32)[y]


if __name__ == "__main__":
    X_mfcc, X_images, X_meta, y = load_multimodal_dataset(max_samples=100)
    print(f"\n  Loaded {len(y)} samples successfully!")