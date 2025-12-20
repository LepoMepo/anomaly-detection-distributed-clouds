from pydantic import BaseModel
from typing import List, Dict


class PredictionRequest(BaseModel):
    feature_name: str
    feature: List[str]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float

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