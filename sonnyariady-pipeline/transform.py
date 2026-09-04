"""Modul Transform TFX: ``preprocessing_fn`` untuk dataset abalone.

Modul ini dijalankan oleh komponen TFX Transform untuk:
1. Membangun vocabulary untuk fitur kategorikal (``Sex``).
2. Menstandarkan (z-score) fitur-fitur numerik.
3. Meneruskan fitur label hasil rekayasa data.

Kode preprocessing yang sama dipakai saat training maupun serving sehingga
tidak terjadi *training-serving skew*.
"""

import os
import sys

import tensorflow as tf
import tensorflow_transform as tft

# Agar modul 'modules' dapat diimpor ketika file ini dimuat oleh executor TFX.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.data_processing import (  # noqa: E402  pylint: disable=wrong-import-position
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERIC_FEATURES,
    transformed_name,
)
from modules.utils import OOV_SIZE, VOCAB_SIZE  # noqa: E402  pylint: disable=wrong-import-position


def preprocessing_fn(inputs):
    """Memetakan fitur mentah menjadi fitur hasil transformasi.

    Args:
        inputs: dict nama fitur mentah -> tensor.

    Returns:
        dict nama fitur hasil transformasi -> tensor.
    """
    outputs = {}

    # Fitur kategorikal -> indeks vocabulary (dengan bucket OOV).
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=VOCAB_SIZE,
            num_oov_buckets=OOV_SIZE,
        )

    # Fitur numerik -> z-score (standarisasi).
    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    # Label diteruskan apa adanya (sudah dibuat pada tahap rekayasa data).
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
