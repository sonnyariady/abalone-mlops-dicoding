"""Arsitektur model & hyperparameter untuk klasifikasi abalon (Keras).

Model dilatih pada fitur hasil transformasi (tf.Transform), sehingga kode
ini dipakai bersama oleh modul Trainer dan Tuner TFX.
"""

import kerastuner as kt
import tensorflow as tf

from modules.data_processing import NUMERIC_FEATURES, transformed_name
from modules.utils import OOV_SIZE, VOCAB_SIZE


def build_keras_model(
    learning_rate: float = 1e-3,
    hidden_units: tuple = (64, 32),
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Membangun DNN untuk klasifikasi biner pada fitur hasil transformasi.

    Arsitektur:
        - Embedding untuk fitur kategorikal ``Sex``.
        - 7 fitur numerik hasil z-score.
        - 2 hidden layer Dense + Dropout.
        - Output sigmoid (probabilitas abalon dewasa).

    Args:
        learning_rate: learning rate optimizer Adam.
        hidden_units: jumlah neuron tiap hidden layer.
        dropout_rate: nilai dropout antar hidden layer.

    Returns:
        ``tf.keras.Model`` yang sudah dikompilasi.
    """
    # Fitur kategorikal Sex -> Embedding (nama input = nama fitur transformasi).
    sex_input = tf.keras.layers.Input(
        shape=(1,), name=transformed_name("Sex"), dtype=tf.int64
    )
    sex_embedding = tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE + OOV_SIZE, output_dim=8, name="sex_embedding"
    )(sex_input)
    sex_embedding = tf.keras.layers.Flatten()(sex_embedding)

    # Fitur numerik hasil z-score.
    numeric_inputs = [
        tf.keras.layers.Input(
            shape=(1,), name=transformed_name(feature), dtype=tf.float32
        )
        for feature in NUMERIC_FEATURES
    ]

    # Gabungkan seluruh fitur menjadi satu vektor.
    concat = tf.keras.layers.Concatenate()([sex_embedding] + numeric_inputs)

    # Hidden layers.
    x = concat
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Output probabilitas kelas dewasa.
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs=[sex_input] + numeric_inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_hyperparam_model(hp: kt.HyperParameters) -> tf.keras.Model:
    """Membangun model dengan ruang pencarian hyperparameter (KerasTuner).

    Hyperparameter yang dicari: learning rate, jumlah unit tiap hidden layer,
    dan nilai dropout.
    """
    learning_rate = hp.Float(
        "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    units_2 = hp.Int("units_2", min_value=8, max_value=64, step=8)
    dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    return build_keras_model(
        learning_rate=learning_rate,
        hidden_units=(units_1, units_2),
        dropout_rate=dropout,
    )
