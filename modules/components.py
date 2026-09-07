"""Modul TFX Components: inisialisasi seluruh komponen machine learning pipeline."""

import os
import tensorflow_model_analysis as tfma
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import (
    LatestBlessedModelResolver,
)
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher,
)

from modules.data_processing import LABEL_KEY


def init_components(
    data_dir: str,
    transform_module: str,
    training_module: str,
    training_steps: int = 5000,
    eval_steps: int = 1000,
    serving_model_dir: str = None,
):
    """Inisialisasi seluruh komponen TFX pipeline end-to-end.

    Komponen yang dibuat:
    1. CsvExampleGen - ingestion dataset CSV dan pembagian train:eval
    2. StatisticsGen - kalkulasi statistik data
    3. SchemaGen - inferensi skema fitur
    4. ExampleValidator - deteksi anomali data
    5. Transform - rekayasa fitur dengan tf.Transform
    6. Trainer - pelatihan model deep learning
    7. Resolver - pemilihan model terbaik (LatestBlessedModelResolver)
    8. Evaluator - evaluasi model menggunakan TFMA
    9. Pusher - publikasi model yang lolos validasi ke direktori serving
    """
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(
        input_base=data_dir,
        output_config=output_config,
    )

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=False,
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module),
        disable_analyzer_cache=True,
    )

    # Batasi steps sesuai ukuran dataset abalone (4177 baris / 64 = ~52 batches train, ~13 batches eval)
    train_args = trainer_pb2.TrainArgs(
        splits=["train"],
        num_steps=min(training_steps, 52),
    )
    eval_args = trainer_pb2.EvalArgs(
        splits=["eval"],
        num_steps=min(eval_steps, 13),
    )

    trainer = Trainer(
        module_file=os.path.abspath(training_module),
        examples=example_gen.outputs["examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=train_args,
        eval_args=eval_args,
    )

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")

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

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
        schema=schema_gen.outputs["schema"],
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=os.path.abspath(serving_model_dir)
            )
        ),
    )

    return (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )
