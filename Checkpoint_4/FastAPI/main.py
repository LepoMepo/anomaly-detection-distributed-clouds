from fastapi import FastAPI, HTTPException, status, Depends, Request
from contextlib import asynccontextmanager
from sqlmodel import Session, select, text

from app.settings.settings import MODEL_PATH, THRESHOLD_PATH
from app.settings.pydantic_models import PredictionRequest, PredictionResponse, HistoryResponse, StatsResponse, DBItems
from app.database.database import Logs, create_db_and_tables, get_session

import joblib

# библиотеки для работы модели
from sklearn.base import BaseEstimator, TransformerMixin
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from app.model.LogTransformer import LogTransformer
import pandas as pd

from time import perf_counter


@asynccontextmanager
async def lifespan(app: FastAPI):
    import __main__
    __main__.LogTransformer = LogTransformer
    try:
        with open(MODEL_PATH, "rb") as f:
            app.state.model = joblib.load(f)
        with open(THRESHOLD_PATH, "rb") as f:
            app.state.threshold = joblib.load(f)
    except Exception:
        app.state.model = None
        app.state.threshold = None

    create_db_and_tables()

    yield


app = FastAPI(lifespan=lifespan)

@app.post("/forward", response_model=PredictionResponse)
def forward_prediction(request: PredictionRequest, session: Session = Depends(get_session)):
    start_time = perf_counter()
    if app.state.model is None or app.state.threshold is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model was not loaded"
        )

    if request.feature_name != "original_message":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request"
        )

    try:
        df = pd.DataFrame({request.feature_name: request.feature})
        prediction = app.state.model["model"].decision_function(app.state.model["transformer"].transform(df))[0]
        prediction_if = (prediction <= app.state.threshold).astype(int)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"модель не смогла обработать данные"
        )

    execution_time = perf_counter() - start_time
    token_count = len(str(request.feature).split())

    db_request = Logs(
        input_data=str(request.feature),
        result="Anomaly" if prediction_if == 1 else "Normal",
        probability=prediction,
        execution_time=execution_time,
        token_count=token_count
    )

    session.add(db_request)
    session.commit()
    session.refresh(db_request)

    return PredictionResponse(
        prediction = "Anomaly" if prediction_if == 1 else "Normal",
        probability = prediction
    )


@app.get("/history", response_model=HistoryResponse)
def get_history(session: Session = Depends(get_session)):
    try:
        statement = select(Logs)
        results = session.exec(statement).all()
        serialized_result = {}
        for log in results:
            serialized_result[log.id] = DBItems(
                timestamp = str(log.timestamp),
                input_data = log.input_data,
                result = log.result,
                probability = log.probability,
                execution_time = log.execution_time,
                token_count = log.token_count
            )
        return HistoryResponse(
            id = serialized_result
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"problem with db"
        )

@app.get('/stats', response_model=StatsResponse)
def get_stats(session: Session = Depends(get_session)):
    try:
        sql_command = text('SELECT execution_time, token_count FROM logs')
        result = session.execute(sql_command).fetchall()
        df_result = pd.DataFrame(result, columns=['execution_time', 'token_count'])
        mean_execution_time = df_result['execution_time'].mean()
        q50 = df_result['execution_time'].quantile(0.5)
        q95 = df_result['execution_time'].quantile(0.95)
        q99 = df_result['execution_time'].quantile(0.99)
        mean_token_size = df_result['token_count'].mean()
        return StatsResponse(
            mean_execution_time = mean_execution_time,
            Q_50 = q50,
            Q_95 = q95,
            Q_99 = q99,
            mean_token_size = mean_token_size
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"problem with stats"
        )