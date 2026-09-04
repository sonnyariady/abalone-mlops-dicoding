"""Modul Tuner TFX: ``tuner_fn`` untuk hyperparameter tuning otomatis.

Menggunakan KerasTuner (RandomSearch) untuk mencari kombinasi learning rate,
jumlah unit hidden layer, dan dropout terbaik berdasarkan metrik
``val_accuracy``.
"""

import os
import sys
from typing import List

import kerastuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.tfxio import dataset_options

# Agar modul 'modules' dapat diimpor ketika file ini dimuat oleh executor TFX.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from modules.data_processing import (  # noqa: E402  pylint: disable=wrong-import-position
    LABEL_KEY,
    transformed_name,
)
from modules.model_building import build_hyperparam_model  # noqa: E402  pylint: disable=wrong-import-position
from modules.utils import set_seed  # noqa: E402  pylint: disable=wrong-import-position

_BATCH_SIZE = 64
_MAX_TRIALS = 5
_TUNING_EPOCHS = 10


def _input_fn(
    file_pattern: List[str],
    data_accessor,
    tf_transform_output: tft.TFTransformOutput,
    batch_size: int = _BATCH_SIZE,
) -> tf.data.Dataset:
    """Membuat dataset tuning dari contoh hasil transformasi."""
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

    return dataset.map(_to_dense)


def tuner_fn(fn_args) -> TunerFnResult:
    """Membangun tuner KerasTuner beserta argumen pencariannya.

    Args:
        fn_args: argumen dari komponen TFX Tuner.

    Returns:
        ``TunerFnResult`` berisi tuner dan ``fit_kwargs`` untuk pencarian.
    """
    set_seed()

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, tf_transform_output
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output
    )

    tuner = kt.RandomSearch(
        build_hyperparam_model,
        objective=kt.Objective("val_accuracy", direction="max"),
        max_trials=_MAX_TRIALS,
        directory=fn_args.working_dir,
        project_name="abalone_hyperparameter_tuning",
    )

    steps_per_epoch = getattr(fn_args, "train_steps", None) or 52
    validation_steps = getattr(fn_args, "eval_steps", None) or 13

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "epochs": _TUNING_EPOCHS,
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "callbacks": [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
        },
    )
