from pathlib import Path

import os

APP_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = APP_DIR / 'model'


MODEL_PATH = MODEL_DIR / 'model.joblib'
THRESHOLD_PATH = MODEL_DIR / 'threshold.joblib'
DRAIN_STATE_PATH =  Path('../model/drain3_state.bin')
DRAIN_CONFIG_PATH = Path('../model/drain3.ini')
DATA_BASE_PATH = APP_DIR


HISTORY_DELETE_TOKEN = os.getenv("HISTORY_DELETE_TOKEN", "TEMPORARY_TOKEN")
