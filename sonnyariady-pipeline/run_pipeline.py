"""Menjalankan pipeline TFX menggunakan Apache Beam (BeamDagRunner).

Cara menjalankan (dari root proyek):

    python sonnyariady-pipeline/run_pipeline.py

Seluruh komponen pipeline (ExampleGen, StatisticsGen, SchemaGen,
ExampleValidator, Transform, Tuner, Trainer, Resolver, Evaluator, Pusher)
diorkestrasikan oleh Apache Beam (DirectRunner) dan hasilnya disimpan di
``pipeline_output/tfx_pipeline`` serta ``serving_model``.
"""

import os
import sys

# Pastikan folder pipeline dan root proyek dapat diimpor.
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PIPELINE_DIR)
for path in (_PIPELINE_DIR, _PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner  # noqa: E402

from configs import (  # noqa: E402
    DATA_ROOT,
    METADATA_PATH,
    PIPELINE_NAME,
    PIPELINE_ROOT,
    SERVING_MODEL_DIR,
)
from pipeline import create_pipeline  # noqa: E402


def main() -> None:
    """Membangun dan menjalankan pipeline dengan Apache Beam."""
    pipeline = create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        serving_model_dir=SERVING_MODEL_DIR,
        metadata_path=METADATA_PATH,
        enable_tuner=True,
    )
    BeamDagRunner().run(pipeline)


if __name__ == "__main__":
    main()
