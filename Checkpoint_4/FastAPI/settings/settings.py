from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = APP_DIR / "model"
DATA_BASE_DIR = APP_DIR / "db_data"

MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.joblib"
DRAIN_STATE_PATH = Path("../model/drain3_state.bin")
DRAIN_CONFIG_PATH = Path("../model/drain3.ini")
