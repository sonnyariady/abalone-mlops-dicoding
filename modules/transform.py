"""Modul Transform TFX: ``preprocessing_fn`` untuk dataset abalone."""

import os
import sys

import tensorflow as tf
import tensorflow_transform as tft

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.data_processing import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERIC_FEATURES,
    transformed_name,
)
from modules.utils import OOV_SIZE, VOCAB_SIZE


def preprocessing_fn(inputs):
    """Memetakan fitur mentah menjadi fitur hasil transformasi."""
    outputs = {}

    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            inputs[feature],
            top_k=VOCAB_SIZE,
            num_oov_buckets=OOV_SIZE,
        )

    for feature in NUMERIC_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
