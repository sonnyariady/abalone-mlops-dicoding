"""Modul Trainer TFX: ``run_fn`` untuk melatih model klasifikasi abalon."""

import os
import sys
from typing import List

import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.tfxio import dataset_options

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.data_processing import (
    LABEL_KEY,
    RINGS_KEY,
    transformed_name,
)
from modules.model_building import build_keras_model
from modules.utils import set_seed

_BATCH_SIZE = 64
_EPOCHS = 15


def _input_fn(
    file_pattern: List[str],
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = _BATCH_SIZE,
) -> tf.data.Dataset:
    """Membuat dataset training/evaluasi dari contoh hasil transformasi."""
    dataset = data_accessor.tf_dataset_factory(
        file_pattern,
        dataset_options.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=transformed_name(LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    )

    def _to_dense(features, label):
        if isinstance(label, tf.SparseTensor):
            label = tf.sparse.to_dense(label)
        label = tf.reshape(label, [-1, 1])
        dense_features = {}
        for k, v in features.items():
            if isinstance(v, tf.SparseTensor):
                dense_features[k] = tf.sparse.to_dense(v, default_value=0)
            else:
                dense_features[k] = v
            dense_features[k] = tf.reshape(dense_features[k], [-1, 1])
        return dense_features, label

    return dataset.map(_to_dense).repeat()


def _get_serve_tf_examples_fn(model: tf.keras.Model, tf_transform_output):
    """Mengembalikan fungsi serving yang menerima serialized tf.Example."""
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ]
    )
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY, None)
        raw_feature_spec.pop(RINGS_KEY, None)
        raw_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    """Melatih model dan mengekspor SavedModel untuk serving."""
    set_seed()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    hp = fn_args.hyperparameters
    hp_values = (hp or {}).get("values", {})
    learning_rate = hp_values.get("learning_rate", 1e-3)
    units_1 = hp_values.get("units_1", 64)
    units_2 = hp_values.get("units_2", 32)
    dropout = hp_values.get("dropout", 0.2)

    model = build_keras_model(
        learning_rate=learning_rate,
        hidden_units=(units_1, units_2),
        dropout_rate=dropout,
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir, update_freq="epoch"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    steps_per_epoch = getattr(fn_args, "train_steps", None) or 52
    validation_steps = getattr(fn_args, "eval_steps", None) or 13

    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        epochs=_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(model, tf_transform_output),
    }
    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
