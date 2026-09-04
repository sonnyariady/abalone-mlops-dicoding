"""Utilitas bersama: konstanta fitur & seed reproduksibilitas."""

import random

import numpy as np
import tensorflow as tf

# Ukuran vocabulary maksimum & jumlah bucket out-of-vocabulary (OOV).
VOCAB_SIZE = 1000
OOV_SIZE = 10

# Seed global agar eksperimen reproducible.
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Menyetel seed untuk random, numpy, dan TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
