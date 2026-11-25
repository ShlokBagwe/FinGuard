from pathlib import Path
import yaml

# Project root is two levels above this file (src/finguard/config.py).
# Anchoring paths here keeps them valid from notebooks, scripts and Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

RAW_DIR = PROJECT_ROOT / config["paths"]["raw_dir"]
INTERIM_DIR = PROJECT_ROOT / config["paths"]["interim_dir"]
PROCESSED_DIR = PROJECT_ROOT / config["paths"]["processed_dir"]
MODELS_DIR = PROJECT_ROOT / config["paths"]["models_dir"]

RAW_FILES = config["raw_files"]
INTERIM_FILES = config["interim_files"]
PROCESSED_FILES = config["processed_files"]

TRAIN_RATIO = config["split"]["train_ratio"]
TIME_COLUMN = config["split"]["time_column"]
