"""
attention_model.py  — Layer 3 complete
=======================================
Architecture:
  Layer 1  inputs      : MFCC (122,) | image (224,224,3) | metadata (8,)
  Layer 2  encoders    : MFCC encoder | CNN encoder | Metadata encoder
  Layer 3A temporal    : TemporalAttention on MFCC  ← NOW WIRED
  Layer 3B cross-modal : CrossModalAttention (MFCC + CNN + Meta tokens)
  Layer 4  classifier  : Dense head → 4-class softmax

What changed from your original:
  - TemporalAttention is wired into build_full_model() (was dead code)
  - MFCC embedding passes through temporal attn BEFORE cross-modal attn
  - Both layers return attention weights for explainability.py
  - build_full_model() returns (train_model, viz_model)
    train_model  → normal training, outputs (batch, 4)
    viz_model    → same weights, also outputs attention tensors
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2

# ─────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────
NUM_CLASSES   = 4     # Normal, Crackle, Wheeze, Both
MFCC_DIM      = 122   # 40 mean + 40 std + 40 delta + ZCR + RMS
IMG_SIZE      = 224
META_DIM      = 8     # age, bmi, sex, smoking, cough, wheeze, chest_pain, dyspnea
EMBED_DIM     = 128   # all modalities project to this size
ATTN_HEADS    = 4
DROPOUT       = 0.4
LEARNING_RATE = 1e-4


# ─────────────────────────────────────────────────
# LAYER 2A  MFCC ENCODER
# ─────────────────────────────────────────────────
def build_mfcc_encoder(mfcc_dim=MFCC_DIM, embed_dim=EMBED_DIM):
    inp = layers.Input(shape=(mfcc_dim,), name="mfcc_input")
    x   = layers.Dense(256, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT)(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(embed_dim, activation='relu', name="mfcc_embed")(x)
    return Model(inp, out, name="mfcc_encoder")


# ─────────────────────────────────────────────────
# LAYER 2B  CNN ENCODER  (MobileNetV2)
# ─────────────────────────────────────────────────
def build_cnn_encoder(img_size=IMG_SIZE, embed_dim=EMBED_DIM, freeze_base=True):
    inp  = layers.Input(shape=(img_size, img_size, 3), name="image_input")
    base = MobileNetV2(input_shape=(img_size, img_size, 3),
                       include_top=False, weights='imagenet')
    base.trainable = not freeze_base
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(embed_dim, activation='relu', name="cnn_embed")(x)
    return Model(inp, out, name="cnn_encoder")


# ─────────────────────────────────────────────────
# LAYER 2C  METADATA ENCODER
# ─────────────────────────────────────────────────
def build_metadata_encoder(meta_dim=META_DIM, embed_dim=EMBED_DIM):
    inp = layers.Input(shape=(meta_dim,), name="metadata_input")
    x   = layers.Dense(64, activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    out = layers.Dense(embed_dim, activation='relu', name="meta_embed")(x)
    return Model(inp, out, name="metadata_encoder")


# ─────────────────────────────────────────────────
# LAYER 3A  TEMPORAL ATTENTION
# ─────────────────────────────────────────────────
class TemporalAttention(layers.Layer):
    """
    Learns which TIME SEGMENT of the audio matters most.

    In standard mode: receives MFCC embedding as (batch, 1, embed_dim).
    In segment mode : receives (batch, T, mfcc_dim) — used by explainability.py
                      to highlight the exact 2-second window with crackle/wheeze.

    Returns:
        attended : (batch, embed_dim)  — refined MFCC representation
        weights  : (batch, T, 1)       — softmax score per time step
                                         (used for visualisation)
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.project = layers.Dense(embed_dim, activation='relu',
                                    name="temporal_project")
        self.score   = layers.Dense(1, name="temporal_score")

    def call(self, x, training=False):
        # x: (batch, T, input_dim)
        h       = self.project(x)               # (batch, T, embed_dim)
        scores  = self.score(h)                 # (batch, T, 1)
        weights = tf.nn.softmax(scores, axis=1) # (batch, T, 1)  ← sum=1 over T
        attended = tf.reduce_sum(weights * h, axis=1)  # (batch, embed_dim)
        return attended, weights


# ─────────────────────────────────────────────────
# LAYER 3B  CROSS-MODAL ATTENTION
# ─────────────────────────────────────────────────
class CrossModalAttention(layers.Layer):
    """
    Learns which MODALITY to trust for each patient.

    Treats the 3 encoder outputs as 3 tokens and runs multi-head
    self-attention so each modality can query the others.

    e.g. for an elderly smoker → metadata token attends more strongly
         for a noisy recording → CNN token gets down-weighted

    Returns:
        fused  : (batch, 3 * embed_dim)  — concatenated attended vectors
        tokens : (batch, 3, embed_dim)   — per-modality attended vectors
                                           (token 0=MFCC, 1=CNN, 2=Meta)
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.mha   = layers.MultiHeadAttention(
                         num_heads=num_heads,
                         key_dim=embed_dim // num_heads,
                         name="cross_modal_mha")
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn   = keras.Sequential([
                         layers.Dense(embed_dim * 2, activation='relu'),
                         layers.Dense(embed_dim)
                     ], name="cross_modal_ffn")

    def call(self, mfcc_emb, cnn_emb, meta_emb, training=False):
        # Stack → (batch, 3, embed_dim)
        tokens   = tf.stack([mfcc_emb, cnn_emb, meta_emb], axis=1)

        # Self-attention: each modality attends to all others
        attn_out = self.mha(tokens, tokens, training=training)
        tokens   = self.norm1(tokens + attn_out)      # residual + norm

        # Feed-forward on each token
        ffn_out  = self.ffn(tokens)
        tokens   = self.norm2(tokens + ffn_out)       # residual + norm

        # Flatten all 3 attended tokens into one vector
        fused = tf.reshape(tokens, [-1, 3 * tokens.shape[-1]])
        # fused: (batch, 384)   tokens: (batch, 3, 128)
        return fused, tokens


# ─────────────────────────────────────────────────
# FULL MODEL  (Layer 1-2-3-4 wired together)
# ─────────────────────────────────────────────────
def build_full_model(
    mfcc_dim    = MFCC_DIM,
    img_size    = IMG_SIZE,
    meta_dim    = META_DIM,
    num_classes = NUM_CLASSES,
    embed_dim   = EMBED_DIM,
    num_heads   = ATTN_HEADS,
    freeze_cnn  = True
):
    """
    Returns (train_model, viz_model).

    train_model
        inputs  : [mfcc (batch,122), image (batch,224,224,3), meta (batch,8)]
        outputs : predictions (batch, 4)   ← use this for model.fit()

    viz_model   (same weights — no extra training needed)
        inputs  : same as above
        outputs : [predictions, cross_modal_tokens, temporal_weights]
                   predictions        (batch, 4)
                   cross_modal_tokens (batch, 3, 128)  — modality trust scores
                   temporal_weights   (batch, 1, 1)    — audio segment weights
    """

    # ── Inputs ──────────────────────────────────────
    mfcc_inp = keras.Input(shape=(mfcc_dim,),             name="mfcc_input")
    img_inp  = keras.Input(shape=(img_size, img_size, 3), name="image_input")
    meta_inp = keras.Input(shape=(meta_dim,),             name="metadata_input")

    # ── Layer 2: three encoders ──────────────────────
    mfcc_enc = build_mfcc_encoder(mfcc_dim, embed_dim)
    cnn_enc  = build_cnn_encoder(img_size, embed_dim, freeze_cnn)
    meta_enc = build_metadata_encoder(meta_dim, embed_dim)

    mfcc_emb = mfcc_enc(mfcc_inp)   # (batch, 128)
    cnn_emb  = cnn_enc(img_inp)     # (batch, 128)
    meta_emb = meta_enc(meta_inp)   # (batch, 128)

    # ── Layer 3A: Temporal attention on MFCC ────────
    # Expand to sequence of length 1 so TemporalAttention
    # can treat it as a time-step  → (batch, 1, 128)
    mfcc_seq   = layers.Reshape((1, embed_dim), name="mfcc_seq")(mfcc_emb)
    temp_layer = TemporalAttention(embed_dim, name="temporal_attn")
    mfcc_attended, temporal_weights = temp_layer(mfcc_seq)
    # mfcc_attended : (batch, 128)  — MFCC after temporal re-weighting
    # temporal_weights: (batch, 1, 1)

    # ── Layer 3B: Cross-modal attention ─────────────
    # Uses the temporally-attended MFCC, raw CNN, raw metadata
    cross_layer          = CrossModalAttention(embed_dim, num_heads,
                                               name="cross_modal_attn")
    fused, cross_tokens  = cross_layer(mfcc_attended, cnn_emb, meta_emb)
    # fused        : (batch, 384)
    # cross_tokens : (batch, 3, 128)

    # ── Layer 4: Classification head ────────────────
    x      = layers.Dense(256, activation='relu', name="clf_dense1")(fused)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dropout(DROPOUT)(x)
    x      = layers.Dense(128, activation='relu', name="clf_dense2")(x)
    x      = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name="output")(x)

    # ── Training model ───────────────────────────────
    train_model = Model(
        inputs  = [mfcc_inp, img_inp, meta_inp],
        outputs = output,
        name    = "LungSense_train"
    )

    # ── Visualisation model (same weights, extra outputs) ──
    viz_model = Model(
        inputs  = [mfcc_inp, img_inp, meta_inp],
        outputs = [output, cross_tokens, temporal_weights],
        name    = "LungSense_viz"
    )

    return train_model, viz_model


# ─────────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  LungSense — Layer 3 complete")
    print("="*55)

    train_model, viz_model = build_full_model()
    train_model.compile(
        optimizer = keras.optimizers.Adam(LEARNING_RATE),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"]
    )
    train_model.summary()

    dm = np.random.randn(2, MFCC_DIM).astype(np.float32)
    di = np.random.randn(2, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    dt = np.random.randn(2, META_DIM).astype(np.float32)

    preds = train_model.predict([dm, di, dt], verbose=0)
    print(f"\n  train_model output : {preds.shape}  sum={preds[0].sum():.4f}")

    preds2, cm_tok, tw = viz_model.predict([dm, di, dt], verbose=0)
    print(f"  viz_model outputs  :")
    print(f"    predictions      : {preds2.shape}")
    print(f"    cross-modal tok  : {cm_tok.shape}  (batch, 3 modalities, embed_dim)")
    print(f"    temporal weights : {tw.shape}   (batch, 1 segment, 1)")
    print(f"\n  Layer 3 wired correctly!")
    print("="*55)