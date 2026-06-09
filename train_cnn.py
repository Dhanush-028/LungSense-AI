import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
SPECTROGRAMS_FOLDER = "spectrograms"     # output of preprocess.py
MODEL_SAVE_PATH     = "cnn_model.h5"
IMG_SIZE            = 224               # MobileNetV2 input size
BATCH_SIZE          = 32
EPOCHS_FROZEN       = 10               # train only top layers first
EPOCHS_FINETUNE     = 10               # then fine-tune last layers
LEARNING_RATE       = 1e-4
FINETUNE_LR         = 1e-5            # lower LR for fine-tuning
CLASSES             = ["Both", "Crackle", "Normal", "Wheeze"]  # sorted alphabetically

print("=" * 58)
print("  CNN Model — MobileNetV2 Transfer Learning")
print("=" * 58)

# ─────────────────────────────────────────
# STEP 1: LOAD IMAGES WITH ImageDataGenerator
# ─────────────────────────────────────────
print("\n[1/5] Loading spectrogram images...")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Training generator — with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,               # normalize pixels 0-1
    validation_split=0.2,         # 80% train, 20% val
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False         # don't flip spectrograms
)

# Validation generator — no augmentation, just rescale
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    SPECTROGRAMS_FOLDER,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    SPECTROGRAMS_FOLDER,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

# Class label mapping
class_indices = train_generator.class_indices
idx_to_class  = {v: k for k, v in class_indices.items()}
num_classes   = len(class_indices)

print(f"\n  Classes found   : {list(class_indices.keys())}")
print(f"  Train images    : {train_generator.samples}")
print(f"  Val   images    : {val_generator.samples}")
print(f"  Batch size      : {BATCH_SIZE}")

# ─────────────────────────────────────────
# STEP 2: BUILD MobileNetV2 MODEL
# ─────────────────────────────────────────
print("\n[2/5] Building MobileNetV2 model...")

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam

# Load MobileNetV2 pretrained on ImageNet — WITHOUT top classifier
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,            # remove ImageNet classifier
    weights='imagenet'            # use pretrained weights
)

# Freeze all base layers — only train our custom head first
base_model.trainable = False

# Add custom classification head for our 4 lung disease classes
x = base_model.output
x = GlobalAveragePooling2D()(x)     # flatten feature maps
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)                 # prevent overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation='softmax')(x)  # 4 classes

model = Model(inputs=base_model.input, outputs=output)

print(f"\n  Base model      : MobileNetV2 (pretrained ImageNet)")
print(f"  Total layers    : {len(model.layers)}")
print(f"  Trainable params (frozen): {model.count_params():,}")

# ─────────────────────────────────────────
# STEP 3: PHASE 1 — TRAIN TOP LAYERS ONLY
# ─────────────────────────────────────────
print(f"\n[3/5] Phase 1 — Training top layers ({EPOCHS_FROZEN} epochs)...")
print("      Base MobileNetV2 is FROZEN — only our head trains\n")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
from tensorflow.keras.callbacks import (EarlyStopping,
                                         ModelCheckpoint,
                                         ReduceLROnPlateau)

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_accuracy', patience=5,
        restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=3, verbose=1)
]

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

best_phase1_acc = max(history1.history['val_accuracy']) * 100
print(f"\n  ✔ Phase 1 best val accuracy: {best_phase1_acc:.2f}%")

# ─────────────────────────────────────────
# STEP 4: PHASE 2 — FINE-TUNE LAST LAYERS
# ─────────────────────────────────────────
print(f"\n[4/5] Phase 2 — Fine-tuning last 30 layers ({EPOCHS_FINETUNE} epochs)...")
print("      Unfreezing last 30 layers of MobileNetV2\n")

# Unfreeze last 30 layers of base model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with very low LR for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

trainable_count = sum(
    1 for l in model.layers if l.trainable)
print(f"  Trainable layers now: {trainable_count}")

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy', patience=6,
        restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.3,
        patience=3, verbose=1)
]

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

best_phase2_acc = max(history2.history['val_accuracy']) * 100
print(f"\n  ✔ Phase 2 best val accuracy: {best_phase2_acc:.2f}%")
print(f"  ✔ Model saved → {MODEL_SAVE_PATH}")

# ─────────────────────────────────────────
# STEP 5: EVALUATE ON VALIDATION SET
# ─────────────────────────────────────────
print("\n[5/5] Final Evaluation:")
print("-" * 58)

from sklearn.metrics import (classification_report,
                              confusion_matrix,
                              accuracy_score)

# Get all predictions
val_generator.reset()
y_pred_proba = model.predict(val_generator, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
y_true       = val_generator.classes

labels = list(idx_to_class[i] for i in sorted(idx_to_class.keys()))
acc    = accuracy_score(y_true, y_pred)

print(f"\n  CNN Final Accuracy : {acc * 100:.2f}%\n")
print(classification_report(
    y_true, y_pred,
    target_names=labels,
    digits=3
))

# Confusion Matrix
print("  Confusion Matrix (rows=Actual, cols=Predicted):\n")
cm = confusion_matrix(y_true, y_pred)
header = "  " + " " * 10 + "".join(f"{l:>10}" for l in labels)
print(header)
for i, row in enumerate(cm):
    print(f"  {labels[i]:<10}" + "".join(f"{v:>10}" for v in row))

# Save class mapping for app.py
import json
with open("cnn_class_indices.json", "w") as f:
    json.dump(class_indices, f)
print("\n  ✔ Class indices saved → cnn_class_indices.json")

print("\n" + "=" * 58)
print(f"  CNN Model accuracy  : {acc * 100:.2f}%")
print("  Next: run compare_models.py to compare CNN vs RF")
print("=" * 58)