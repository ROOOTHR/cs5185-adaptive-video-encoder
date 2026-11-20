"""Paths, dataset partitions, and the fixed experiment budget."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DEFAULT = Path(r"D:\gcg\Multimedia\project\CS5185_Project_Video_Dataset")
RAW_DIR = Path(os.environ.get("CS5185_DATASET", str(DATASET_DEFAULT))).resolve()
META_DIR = RAW_DIR / "metadata"
RESULT_DIR = ROOT / "results"
RUN_DIR = RESULT_DIR / "runs"
CHECK_DIR = RESULT_DIR / "checks"

MB = 1_000_000
OVERHEAD = 0.98
RATIOS = (0.05, 0.02)
MAX_ATTEMPTS = 4
MAX_ADJUST = 0.30
ENCODE_THREADS = 4

DEV_IDS = ("01_FourPeople", "02_Beauty", "03_Bosphorus", "04_HoneyBee",
           "07_Mobcal", "08_Shields")
VAL_IDS = ("11_Sintel", "12_ControlledBurn")
TEST_IDS = ("05_Jockey", "06_ShakeNDry", "09_Parkrun", "10_Stockholm")
ALL_IDS = tuple(sorted(DEV_IDS + VAL_IDS + TEST_IDS))


def ensure_dirs() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
