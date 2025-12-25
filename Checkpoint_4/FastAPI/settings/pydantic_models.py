from pydantic import BaseModel
from typing import List, Dict, Optional


# Модели для аутентификации
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Модели для предсказаний с аутентификацией
class PredictionRequest(BaseModel):
    feature_name: str
    feature: List[str]
    # Опциональные поля для аутентификации (при первом запросе)
    username: Optional[str] = None
    password: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    # Токены возвращаются при успешной аутентификации
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_type: Optional[str] = None


class DBItems(BaseModel):
    timestamp: str
    input_data: str
    result: str
    probability: float
    execution_time: float
    token_count: int


class HistoryResponse(BaseModel):
    id: Dict[int, DBItems]


class StatsResponse(BaseModel):
    mean_execution_time: float
    Q_50: float
    Q_95: float
    Q_99: float
    mean_token_size: float
