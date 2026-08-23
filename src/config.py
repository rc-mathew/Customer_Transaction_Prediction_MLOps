from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"


# ---------------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------------

TARGET_COLUMN = "target"
ID_COLUMN = "ID_code"

EXPECTED_FEATURE_COUNT = 200


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20