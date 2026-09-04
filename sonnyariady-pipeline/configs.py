"""Konfigurasi terpusat untuk pipeline TFX Proyek Akhir MLOps.

Seluruh path, nama pipeline, dan parameter komponen didefinisikan di satu
tempat agar mudah dipelihara (menerapkan prinsip clean code - DRY).
"""

import os

# Identitas peserta (username Dicoding).
USERNAME = "sonnyariady"

# Akar direktori proyek (dua level di atas file ini).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Dataset ---
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
DATA_FILE = os.path.join(DATA_ROOT, "abalone.csv")

# --- Pipeline ---
PIPELINE_NAME = "abalone_mlops_pipeline"
PIPELINE_ROOT = os.path.join(PROJECT_ROOT, "pipeline_output", "tfx_pipeline")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

# --- Direktori modul pipeline (folder <username>-pipeline) ---
MODULE_ROOT = os.path.join(PROJECT_ROOT, "sonnyariady-pipeline")
TRANSFORM_MODULE = os.path.join(MODULE_ROOT, "transform.py")
TRAINER_MODULE = os.path.join(MODULE_ROOT, "trainer.py")
TUNER_MODULE = os.path.join(MODULE_ROOT, "tuner.py")

# --- Serving model (output komponen Pusher) ---
SERVING_MODEL_DIR = os.path.join(PROJECT_ROOT, "serving_model")

# --- Pembagian data train/eval ---
TRAIN_SPLIT = 0.8
EVAL_SPLIT = 1.0 - TRAIN_SPLIT
