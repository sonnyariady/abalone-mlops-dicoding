"""Merakit seluruh komponen menjadi sebuah TFX Pipeline (DAG)."""

from typing import List, Optional

from tfx import v1 as tfx
from tfx.orchestration import metadata
from tfx.proto import trainer_pb2

from components import (
    create_evaluator,
    create_example_gen,
    create_example_validator,
    create_pusher,
    create_resolver,
    create_schema_gen,
    create_statistics_gen,
    create_trainer,
    create_transform,
    create_tuner,
)
from configs import DATA_ROOT, METADATA_PATH, PIPELINE_NAME, PIPELINE_ROOT, SERVING_MODEL_DIR


def create_pipeline(
    pipeline_name: str = PIPELINE_NAME,
    pipeline_root: str = PIPELINE_ROOT,
    data_root: str = DATA_ROOT,
    serving_model_dir: str = SERVING_MODEL_DIR,
    metadata_path: Optional[str] = METADATA_PATH,
    enable_tuner: bool = True,
) -> tfx.dsl.Pipeline:
    """Membangun DAG pipeline TFX end-to-end.

    Urutan komponen:
        ExampleGen -> StatisticsGen -> SchemaGen -> ExampleValidator
        -> Transform -> Tuner -> Trainer -> Resolver -> Evaluator -> Pusher

    Args:
        pipeline_name: nama pipeline.
        pipeline_root: direktori penyimpanan artifact pipeline.
        data_root: direktori dataset CSV.
        serving_model_dir: direktori output model serving (Pusher).
        metadata_path: path file metadata MLMD (SQLite).
        enable_tuner: apakah komponen Tuner diikutsertakan.

    Returns:
        Objek ``tfx.dsl.Pipeline`` siap dijalankan dengan Apache Beam.
    """
    example_gen = create_example_gen(data_root)
    statistics_gen = create_statistics_gen(example_gen.outputs["examples"])
    schema_gen = create_schema_gen(statistics_gen.outputs["statistics"])
    example_validator = create_example_validator(
        statistics_gen.outputs["statistics"], schema_gen.outputs["schema"]
    )
    transform = create_transform(
        example_gen.outputs["examples"], schema_gen.outputs["schema"]
    )

    train_args = trainer_pb2.TrainArgs(splits=["train"], num_steps=52)
    eval_args = trainer_pb2.EvalArgs(splits=["eval"], num_steps=13)

    tuner = None
    if enable_tuner:
        tuner = create_tuner(
            example_gen.outputs["examples"],
            transform.outputs["transform_graph"],
            train_args,
            eval_args,
        )

    trainer = create_trainer(
        example_gen.outputs["examples"],
        transform.outputs["transform_graph"],
        schema_gen.outputs["schema"],
        train_args,
        eval_args,
        tuner=tuner,
    )
    resolver = create_resolver()
    evaluator = create_evaluator(
        example_gen.outputs["examples"],
        trainer,
        resolver,
        schema_gen.outputs["schema"],
    )
    pusher = create_pusher(trainer, evaluator, serving_model_dir)

    components: List[tfx.dsl.components.base.base_component.BaseComponent] = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
    ]
    if tuner is not None:
        components.append(tuner)
    components.extend([trainer, resolver, evaluator, pusher])

    connection_config = None
    if metadata_path is not None:
        connection_config = metadata.sqlite_metadata_connection_config(metadata_path)

    # Cache sengaja dimatikan agar seluruh komponen selalu dieksekusi penuh
    # (artifacts deterministik untuk review) dan untuk menghindari keterbatasan
    # MLMD filtering pada platform Windows (ZetaSQL belum terkompilasi di
    # Windows sehingga cache lookup dengan filter query tidak didukung).
    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=connection_config,
    )
