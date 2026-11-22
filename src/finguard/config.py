from pathlib import Path
import yaml

# Step 1: find the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Step 2: read config.yaml
with open(PROJECT_ROOT / "config.yaml") as f:
    config = yaml.safe_load(f)

# Step 3: turn the folder names into full paths
RAW_DIR = PROJECT_ROOT / config["paths"]["raw_dir"]
INTERIM_DIR = PROJECT_ROOT / config["paths"]["interim_dir"]
PROCESSED_DIR = PROJECT_ROOT / config["paths"]["processed_dir"]
MODELS_DIR = PROJECT_ROOT / config["paths"]["models_dir"]

# Step 4: file names and settings
RAW_FILES = config["raw_files"]
INTERIM_FILES = config["interim_files"]
PROCESSED_FILES = config["processed_files"]
TRAIN_RATIO = config["split"]["train_ratio"]
TIME_COLUMN = config["split"]["time_column"]
