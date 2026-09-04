"""Fungsi pembuat seluruh komponen TFX pipeline (prinsip clean code).

Setiap komponen dibuat melalui fungsi khusus dengan parameter eksplisit
sehingga pipeline mudah dibaca, diuji, dan digunakan ulang - baik dari
notebook, dari ``run_pipeline.py``, maupun dari ``InteractiveContext``.
"""

from typing import Optional

import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.proto import example_gen_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

from configs import (
    TRAINER_MODULE,
    TRANSFORM_MODULE,
    TUNER_MODULE,
)
from modules.data_processing import LABEL_KEY


def create_example_gen(data_root: str) -> tfx.components.CsvExampleGen:
    """Membuat komponen ExampleGen: membaca CSV & membagi train:eval = 80:20.

    Pembagian dilakukan dengan hash bucket pada baris data sehingga hasilnya
    deterministik.
    """
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    return tfx.components.CsvExampleGen(
        input_base=data_root, output_config=output_config
    )


def create_statistics_gen(examples) -> tfx.components.StatisticsGen:
    """Membuat komponen StatisticsGen: menghitung statistik data tiap split."""
    return tfx.components.StatisticsGen(examples=examples)


def create_schema_gen(statistics) -> tfx.components.SchemaGen:
    """Membuat komponen SchemaGen: inferensi skema data dari statistik."""
    return tfx.components.SchemaGen(statistics=statistics, infer_feature_shape=False)


def create_example_validator(statistics, schema) -> tfx.components.ExampleValidator:
    """Membuat komponen ExampleValidator: deteksi anomali data terhadap skema."""
    return tfx.components.ExampleValidator(statistics=statistics, schema=schema)


def create_transform(examples, schema) -> tfx.components.Transform:
    """Membuat komponen Transform: rekayasa fitur dengan tf.Transform."""
    return tfx.components.Transform(
        examples=examples,
        schema=schema,
        module_file=TRANSFORM_MODULE,
        disable_analyzer_cache=True,
    )


def create_tuner(
    examples, transform_graph, train_args, eval_args
) -> tfx.components.Tuner:
    """Membuat komponen Tuner: hyperparameter tuning otomatis (KerasTuner)."""
    return tfx.components.Tuner(
        module_file=TUNER_MODULE,
        examples=examples,
        transform_graph=transform_graph,
        train_args=train_args,
        eval_args=eval_args,
    )


def create_trainer(
    examples,
    transform_graph,
    schema,
    train_args,
    eval_args,
    tuner: Optional[tfx.components.Tuner] = None,
) -> tfx.components.Trainer:
    """Membuat komponen Trainer: melatih model Keras pada fitur transformasi.

    Jika ``tuner`` diberikan, hyperparameter terbaik hasil tuning otomatis
    digunakan untuk training.
    """
    kwargs = {
        "module_file": TRAINER_MODULE,
        "examples": examples,
        "transform_graph": transform_graph,
        "schema": schema,
        "train_args": train_args,
        "eval_args": eval_args,
    }
    if tuner is not None:
        kwargs["hyperparameters"] = tuner.outputs["best_hyperparameters"]
    return tfx.components.Trainer(**kwargs)


def create_resolver() -> Resolver:
    """Membuat komponen Resolver: memilih model terbaik sebelumnya (latest blessed)."""
    return Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")


def create_evaluator(examples, trainer, resolver, schema) -> tfx.components.Evaluator:
    """Membuat komponen Evaluator: evaluasi performa model dengan TFMA.

    Metrik yang dihitung: ExampleCount, BinaryAccuracy, AUC, Precision, Recall.
    """
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key=LABEL_KEY)],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=2,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                ]
            )
        ],
        slicing_specs=[tfma.SlicingSpec()],
    )
    return tfx.components.Evaluator(
        examples=examples,
        model=trainer.outputs["model"],
        baseline_model=resolver.outputs["model"],
        eval_config=eval_config,
        schema=schema,
    )


def create_pusher(trainer, evaluator, serving_model_dir: str) -> tfx.components.Pusher:
    """Membuat komponen Pusher: mem-publish model terbaik ke direktori serving."""
    return tfx.components.Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )
