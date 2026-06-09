"""
train_attention.py  — works with Layer 3 complete model
=========================================================
Two-phase training:
  Phase 1 (10 epochs) : CNN frozen, train MFCC + metadata + attention layers
  Phase 2 (10 epochs) : Unfreeze last 30 CNN layers, fine-tune everything

Val split bug from original is fixed here too.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

from attention_model    import build_full_model, NUM_CLASSES
from multimodal_dataset import load_multimodal_dataset, to_categorical, LABEL_MAP

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS_PHASE1   = 10
EPOCHS_PHASE2   = 10
MODEL_SAVE      = "attention_model.h5"
VIZ_MODEL_SAVE  = "attention_viz_model.h5"
SCALER_SAVE     = "attention_mfcc_scaler.pkl"
HISTORY_SAVE    = "training_history.npy"

print("="*55)
print("  LungSense — Attention Training (Layer 3 complete)")
print("="*55)

# ─────────────────────────────────────────────────
# STEP 1  LOAD DATA
# ─────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
X_mfcc, X_images, X_meta, y_int = load_multimodal_dataset()
y_cat = to_categorical(y_int, NUM_CLASSES)

# ─────────────────────────────────────────────────
# STEP 2  PREPROCESS + SPLIT  (val split bug fixed)
# ─────────────────────────────────────────────────
print("\n[2/5] Preprocessing and splitting 70/15/15...")

scaler    = StandardScaler()
X_mfcc_sc = scaler.fit_transform(X_mfcc)
joblib.dump(scaler, SCALER_SAVE)
print(f"  Scaler saved → {SCALER_SAVE}")

# First cut: 70% train, 30% temp
(X_m_tr,  X_m_tmp,
 X_i_tr,  X_i_tmp,
 X_mt_tr, X_mt_tmp,
 y_tr,    y_tmp,
 y_int_tr, y_int_tmp) = train_test_split(
    X_mfcc_sc, X_images, X_meta, y_cat, y_int,
    test_size=0.30, random_state=42, stratify=y_int
)

# Second cut: split temp 50/50 → val and test
half = len(y_tmp) // 2
X_m_val,  X_m_test  = X_m_tmp[:half],  X_m_tmp[half:]
X_i_val,  X_i_test  = X_i_tmp[:half],  X_i_tmp[half:]
X_mt_val, X_mt_test = X_mt_tmp[:half], X_mt_tmp[half:]
y_val,    y_test    = y_tmp[:half],    y_tmp[half:]

print(f"  Train : {len(y_tr)}")
print(f"  Val   : {len(y_val)}")
print(f"  Test  : {len(y_test)}")

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=np.argmax(y_tr, axis=1)
)
cw = dict(enumerate(class_weights))
print(f"  Class weights: { {k: round(v,2) for k,v in cw.items()} }")

# ─────────────────────────────────────────────────
# STEP 3  BUILD MODEL
# ─────────────────────────────────────────────────
print("\n[3/5] Building model (Layer 3 complete)...")
train_model, viz_model = build_full_model(freeze_cnn=True)
train_model.compile(
    optimizer = keras.optimizers.Adam(1e-4),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)
print(f"  Parameters: {train_model.count_params():,}")

# ─────────────────────────────────────────────────
# STEP 4  PHASE 1 — frozen CNN
# ─────────────────────────────────────────────────
print(f"\n[4/5] Phase 1 — frozen CNN  ({EPOCHS_PHASE1} epochs)...")

cb1 = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                                    save_best_only=True, verbose=1),
]

h1 = train_model.fit(
    x=[X_m_tr, X_i_tr, X_mt_tr], y=y_tr,
    validation_data=([X_m_val, X_i_val, X_mt_val], y_val),
    batch_size=BATCH_SIZE, epochs=EPOCHS_PHASE1,
    class_weight=cw, callbacks=cb1, verbose=1
)
best_p1 = max(h1.history['val_accuracy']) * 100
print(f"\n  Phase 1 best val acc: {best_p1:.2f}%")

# ─────────────────────────────────────────────────
# STEP 5  PHASE 2 — unfreeze CNN last 30 layers
# ─────────────────────────────────────────────────
print(f"\n[5/5] Phase 2 — unfreeze CNN last 30 layers  ({EPOCHS_PHASE2} epochs)...")

cnn_encoder = train_model.get_layer("cnn_encoder")
base_model  = None
for layer in cnn_encoder.layers:
    if 'mobilenetv2' in layer.name.lower():
        base_model = layer
        break

if base_model:
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    print(f"  Unfroze last 30 layers of MobileNetV2")
else:
    print("  WARNING: MobileNetV2 base not found — running phase 2 without CNN unfreeze")

train_model.compile(
    optimizer = keras.optimizers.Adam(1e-5),   # lower LR for fine-tuning
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

cb2 = [
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                      patience=3, min_lr=1e-8, verbose=1),
    keras.callbacks.ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                                    save_best_only=True, verbose=1),
]

h2 = train_model.fit(
    x=[X_m_tr, X_i_tr, X_mt_tr], y=y_tr,
    validation_data=([X_m_val, X_i_val, X_mt_val], y_val),
    batch_size=BATCH_SIZE, epochs=EPOCHS_PHASE2,
    class_weight=cw, callbacks=cb2, verbose=1
)
best_p2 = max(h2.history['val_accuracy']) * 100
print(f"\n  Phase 2 best val acc: {best_p2:.2f}%")

# ─────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────
print("\n" + "="*55)
print("  Test set evaluation")
print("="*55)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred_prob = train_model.predict([X_m_test, X_i_test, X_mt_test], verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)
y_true      = np.argmax(y_test, axis=1)
classes     = list(LABEL_MAP.keys())
acc         = accuracy_score(y_true, y_pred)

print(f"\n  Test accuracy : {acc*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=classes, digits=3))

cm = confusion_matrix(y_true, y_pred)
print("  Confusion matrix:")
print("  " + " "*12 + "".join(f"{c:>10}" for c in classes))
for i, row in enumerate(cm):
    print(f"  {classes[i]:<12}" + "".join(f"{v:>10}" for v in row))

# ─────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────
viz_model.save(VIZ_MODEL_SAVE)

# Merge phase histories for easy plotting
merged = {
    'accuracy':      h1.history['accuracy']     + h2.history['accuracy'],
    'val_accuracy':  h1.history['val_accuracy'] + h2.history['val_accuracy'],
    'loss':          h1.history['loss']         + h2.history['loss'],
    'val_loss':      h1.history['val_loss']     + h2.history['val_loss'],
    'phase_boundary': len(h1.history['accuracy']),
    'test_accuracy': acc
}
np.save(HISTORY_SAVE, merged)

print(f"\n  train model  → {MODEL_SAVE}")
print(f"  viz model    → {VIZ_MODEL_SAVE}")
print(f"  history      → {HISTORY_SAVE}")
print("="*55)
print(f"  Phase 1 : {best_p1:.2f}%")
print(f"  Phase 2 : {best_p2:.2f}%")
print(f"  Test    : {acc*100:.2f}%")
print("="*55)