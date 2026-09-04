"""Rekayasa data untuk dataset abalone.

Bertanggung jawab untuk:
1. Mengunduh dataset mentah abalone dari UCI Machine Learning Repository.
2. Menambahkan nama kolom.
3. Membuat fitur label biner (klasifikasi usia abalon).
4. Menyimpan dataset akhir dalam format CSV untuk dibaca komponen ExampleGen.
"""

import os
import urllib.request
from typing import List

import pandas as pd

# Nama kolom dataset mentah abalone (sumber: UCI Machine Learning Repository).
COLUMN_NAMES: List[str] = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]

# Fitur numerik yang akan distandarkan (z-score) pada tahap Transform.
NUMERIC_FEATURES: List[str] = [
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
]

# Fitur kategorikal yang akan di-encode dengan vocabulary.
CATEGORICAL_FEATURES: List[str] = ["Sex"]

# Label hasil rekayasa data (1 = dewasa, 0 = muda).
LABEL_KEY = "label"
RINGS_KEY = "Rings"

# URL sumber data: Abalone Data Set, UCI Machine Learning Repository.
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur setelah proses transformasi.

    Contoh: "Length" -> "Length_xf".
    """
    return key + "_xf"


def _download_raw_rows() -> List[str]:
    """Mengunduh seluruh baris data mentah abalone dari UCI (tanpa header)."""
    with urllib.request.urlopen(DATA_URL) as response:  # noqa: S310
        rows = response.read().decode("utf-8").splitlines()
    return [row for row in rows if row.strip()]


def prepare_dataset(dest_dir: str, label_threshold: int = 9, seed: int = 42) -> str:
    """Mempersiapkan dataset akhir dengan fitur label biner.

    Data mentah diunduh, diberi nama kolom, lalu dibuat fitur label:
    label = 1 jika jumlah cincin (Rings) > ``label_threshold`` (abalon dewasa),
    selain itu 0 (abalon muda). Baris diacak dengan seed tetap agar hasil
    reproducible.

    Args:
        dest_dir: direktori tujuan penyimpanan (berisi hanya ``abalone.csv``
            agar dapat langsung dibaca komponen ExampleGen).
        label_threshold: ambang jumlah cincin untuk label dewasa/muda.
        seed: seed pengacakan baris.

    Returns:
        Path file CSV siap dipakai pipeline (``data/abalone.csv``).
    """
    prepared_path = os.path.join(dest_dir, "abalone.csv")
    if os.path.exists(prepared_path):
        return prepared_path

    os.makedirs(dest_dir, exist_ok=True)
    raw_rows = _download_raw_rows()

    frame = pd.DataFrame([row.split(",") for row in raw_rows], columns=COLUMN_NAMES)
    for column in NUMERIC_FEATURES + [RINGS_KEY]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[LABEL_KEY] = (frame[RINGS_KEY] > label_threshold).astype("int64")
    frame = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    frame.to_csv(prepared_path, index=False)
    return prepared_path
