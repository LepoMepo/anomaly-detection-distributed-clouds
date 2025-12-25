from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

import os

APP_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = APP_DIR / "model"
DATA_BASE_DIR = APP_DIR / "db_data"

MODEL_PATH = MODEL_DIR / "model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.joblib"
DRAIN_STATE_PATH = Path("../model/drain3_state.bin")
DRAIN_CONFIG_PATH = Path("../model/drain3.ini")

class Settings(BaseSettings):
    jwt_secret_key: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: str = "30"
    jwt_refresh_token_expire_days: str = "7"
    history_delete_token: SecretStr

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )
