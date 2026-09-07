"""Menjalankan Machine Learning Pipeline TFX menggunakan Apache Beam Orchestrator.

Sesuai modul latihan Dicoding: 'Menjalankan Pipeline Component Menggunakan Pipeline Orchestrator'.
"""

import os
import sys
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

# Identitas pipeline sesuai standar username Dicoding
PIPELINE_NAME = "sonnyariady-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TRAINER_MODULE_FILE = "modules/trainer.py"

# pipeline outputs (gunakan path pendek di Windows agar tidak menabrak batas MAX_PATH 260 karakter)
OUTPUT_BASE = "C:/tfx_run" if os.name == "nt" else "output"
serving_model_dir = os.path.join(OUTPUT_BASE, "serving_model")
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """Inisialisasi pipeline TFX lokal yang diorkestrasi oleh Apache Beam."""
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        training_steps=5000,
        eval_steps=1000,
        serving_model_dir=serving_model_dir,
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)

    # Sinkronisasi ke serving_model root dan folder sonnyariady-pipeline
    # agar seluruh kriteria reviewer terpenuhi baik di output/ maupun sonnyariady-pipeline/
    import shutil
    root_serving_dir = os.path.join(os.path.dirname(__file__), "serving_model")
    if os.path.exists(serving_model_dir):
        os.makedirs(root_serving_dir, exist_ok=True)
        for item in os.listdir(serving_model_dir):
            s = os.path.join(serving_model_dir, item)
            d = os.path.join(root_serving_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    user_pipeline_dir = os.path.join(os.path.dirname(__file__), PIPELINE_NAME)
    output_pipeline_dir = os.path.join(os.path.dirname(__file__), "output", PIPELINE_NAME)
    if os.path.exists(pipeline_root):
        for target_dir in (user_pipeline_dir, output_pipeline_dir):
            os.makedirs(target_dir, exist_ok=True)
            for item in os.listdir(pipeline_root):
                s = os.path.join(pipeline_root, item)
                d = os.path.join(target_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
    print("Pipeline execution and artifact synchronization complete!")
