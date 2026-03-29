from fastapi import FastAPI, HTTPException, status, Depends, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from contextlib import asynccontextmanager
from sqlmodel import Session, select, text
from typing import Optional
import pickle
from pathlib import Path
import os

from settings.settings import (
    IF_MODEL_PATH,
    IF_THRESHOLD_PATH,
    LSTM_MODEL_PATH,
    LSTM_TRANSFORMER_PATH,
    Settings,
)
from settings.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HistoryResponse,
    StatsResponse,
    DBItems,
    TokenResponse,
    RefreshTokenRequest,
)
from database.database import Logs, create_db_and_tables, get_session
from auth.jwt_handler import (
    authenticate_user,
    create_tokens,
    verify_token,
    refresh_access_token,
)

import joblib

# библиотеки для работы модели
from sklearn.base import BaseEstimator, TransformerMixin
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
from model.LogTransformer import LogTransformer
from model.SequenceTransformer import SequenceTransformer
import pandas as pd

from time import perf_counter


# Схема безопасности для Bearer токена
security = HTTPBearer(auto_error=False)


# TO DO временный костыль для загрузки модели
class UniversalUnpickler(pickle.Unpickler):
    """Кастомный Unpickler, который заменяет WindowsPath на Path"""

    def find_class(self, module, name):
        if module == "pathlib" and name == "WindowsPath":
            return Path  # Заменяем WindowsPath на обычный Path
        return super().find_class(module, name)


# TO DO временный костыль для загрузки модели
def load_model_safely(model_path):
    """Безопасная загрузка модели с WindowsPath"""
    model_path = Path(model_path)

    if os.name == 'nt':
        return joblib.load(model_path)

    # Способ 1: Через кастомный unpickler
    try:
        with open(model_path, "rb") as f:
            unpickler = UniversalUnpickler(f)
            model = unpickler.load()
        print(f"✓ Model loaded with UniversalUnpickler: {model_path.name}")
        return model
    except Exception as e1:
        print(f"UniversalUnpickler failed: {e1}")

    # Способ 2: Через joblib с monkey-patching
    try:
        import pathlib

        # Временно подменяем WindowsPath
        original_windows_path = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.Path

        model = joblib.load(model_path)

        # Восстанавливаем
        pathlib.WindowsPath = original_windows_path

        print(f"✓ Model loaded with monkey-patch: {model_path.name}")
        return model
    except Exception as e2:
        print(f"Joblib monkey-patch failed: {e2}")

    # Способ 3: Загружаем как сырые байты и заменяем
    try:
        with open(model_path, "rb") as f:
            data = f.read()

        # Заменяем WindowsPath на Path в бинарных данных
        data = data.replace(b"WindowsPath", b"Path")

        import io

        model = pickle.load(io.BytesIO(data))
        print(f"✓ Model loaded with bytes replacement: {model_path.name}")
        return model
    except Exception as e3:
        print(f"Bytes replacement failed: {e3}")

    raise ValueError(f"Failed to load model {model_path}")


def load_if_bundle():
    model = load_model_safely(IF_MODEL_PATH)
    threshold = load_model_safely(IF_THRESHOLD_PATH)
    return model, threshold


def load_lstm_bundle():
    if not LSTM_MODEL_PATH.exists():
        return None
    try:
        import torch
    except Exception as e:
        print(f"PyTorch not available: {e}")
        return None

    checkpoint = torch.load(LSTM_MODEL_PATH, map_location="cpu")
    bundle = {"model": None, "config": {}, "transformer": None}

    transformer = None
    if LSTM_TRANSFORMER_PATH.exists():
        transformer = joblib.load(LSTM_TRANSFORMER_PATH)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        vocab_size = checkpoint.get("vocab_size") or config.get("vocab_size")
        if vocab_size is None and transformer is not None:
            vocab_size = len(transformer.template_list_ or [])
        if vocab_size is None:
            raise ValueError("LSTM checkpoint missing vocab_size")
        embedding_dim = config.get("embedding_dim", 32)
        hidden_size = config.get("hidden_size", 64)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.0)
        from model.lstm_model import LSTMNextEventModel

        model = LSTMNextEventModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
        model.load_state_dict(checkpoint["state_dict"])
        bundle["model"] = model
        bundle["config"] = config
    elif isinstance(checkpoint, torch.nn.Module):
        bundle["model"] = checkpoint
        config = getattr(checkpoint, "config", {})
        bundle["config"] = config if isinstance(config, dict) else {}
    else:
        raise ValueError("Unsupported LSTM checkpoint format")

    bundle["model"].eval()

    bundle["transformer"] = transformer
    return bundle


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """Получение текущего пользователя из токена (опционально)"""
    if credentials is None:
        return None

    token = credentials.credentials
    username = verify_token(token, token_type="access")
    return username


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
) -> str:
    """Обязательная аутентификация для защищенных эндпоинтов"""
    token = credentials.credentials
    username = verify_token(token, token_type="access")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


def _authenticate_request(
    request: PredictionRequest, current_user: Optional[str]
) -> tuple[str, Optional[dict]]:
    tokens = None
    if current_user is None:
        if request.username and request.password:
            user = authenticate_user(request.username, request.password)
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid username or password",
                )
            current_user = request.username
            tokens = create_tokens(current_user)
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Provide username/password or Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return current_user, tokens


@asynccontextmanager
async def lifespan(app: FastAPI):
    import __main__

    __main__.LogTransformer = LogTransformer
    __main__.SequenceTransformer = SequenceTransformer

    try:
        # with open(MODEL_PATH, "rb") as f:
        #     app.state.model = joblib.load(f)
        # with open(THRESHOLD_PATH, "rb") as f:
        #     app.state.threshold = joblib.load(f)

        # TO DO временный костыль для загрузки модели
        app.state.if_model, app.state.if_threshold = load_model_safely(IF_MODEL_PATH), load_model_safely(IF_THRESHOLD_PATH)
        app.state.lstm_bundle = load_lstm_bundle()
        settings = Settings()
        app.state.history_delete_token = settings.history_delete_token.get_secret_value()

    except Exception as e:
        app.state.if_model = None
        app.state.if_threshold = None
        app.state.lstm_bundle = None
        print(f"Error loading model: {repr(e)}")
    await create_db_and_tables()

    yield


app = FastAPI(
    lifespan=lifespan,
    title="Anomaly Detection API",
    description="API для обнаружения аномалий в логах с JWT аутентификацией",
    version="0.1.0",
)



@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse("bad request", status_code=400)


@app.post("/forward", response_model=PredictionResponse)
async def forward_prediction(
    request: PredictionRequest,
    session: Session = Depends(get_session),
    current_user: Optional[str] = Depends(get_current_user),
):
    return await forward_prediction_if(request, session, current_user)


@app.post("/forward/if", response_model=PredictionResponse)
async def forward_prediction_if(
    request: PredictionRequest,
    session: Session = Depends(get_session),
    current_user: Optional[str] = Depends(get_current_user),
):
    """
    Предсказание аномалий в логах.

    Аутентификация может быть выполнена двумя способами:
    1. Передать username и password в теле запроса (вернёт токены в ответе)
    2. Передать Bearer token в заголовке Authorization
    """
    start_time = perf_counter()

    # Если нет токена в заголовке, проверяем логин/пароль в теле запроса
    current_user, tokens = _authenticate_request(request, current_user)

    if app.state.if_model is None or app.state.if_threshold is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="IF model was not loaded",
        )

    if request.feature_name != "original_message":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="bad request"
        )

    try:
        df = pd.DataFrame({request.feature_name: request.feature})
        prediction = app.state.model["model"].decision_function(app.state.model["transformer"].transform(df))[0]
        prediction_if = (prediction <= app.state.if_threshold).astype(int)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"модель не смогла обработать данные",
        )

    execution_time = perf_counter() - start_time
    token_count = len(str(request.feature).split())

    db_request = Logs(
        input_data=str(request.feature),
        result="Anomaly" if prediction_if == 1 else "Normal",
        probability=prediction,
        execution_time=execution_time,
        token_count=token_count,
    )

    session.add(db_request)
    await session.commit()
    await session.refresh(db_request)

    response = PredictionResponse(
        prediction="Anomaly" if prediction_if == 1 else "Normal", probability=prediction
    )

    # Если были созданы токены при аутентификации через логин/пароль
    if tokens:
        response.access_token = tokens["access_token"]
        response.refresh_token = tokens["refresh_token"]
        response.token_type = tokens["token_type"]

    return response


@app.post("/forward/lstm", response_model=PredictionResponse)
async def forward_prediction_lstm(
    request: PredictionRequest,
    session: Session = Depends(get_session),
    current_user: Optional[str] = Depends(get_current_user),
):
    start_time = perf_counter()

    # Если нет токена в заголовке, проверяем логин/пароль в теле запроса
    current_user, tokens = _authenticate_request(request, current_user)

    if app.state.lstm_bundle is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LSTM model was not loaded",
        )

    if request.feature_name != "original_message":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="bad request"
        )

    transformer = None
    if isinstance(app.state.lstm_bundle, dict):
        transformer = app.state.lstm_bundle.get("transformer")

    if transformer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LSTM transformer was not loaded",
        )

    df = pd.DataFrame({request.feature_name: request.feature})
    windows_df = transformer.transform(df)
    if windows_df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid windows for LSTM inference",
        )
    if "target" not in windows_df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="LSTM windows must include targets for scoring",
        )

    try:
        import torch
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PyTorch not available: {e}",
        )

    model = None
    config = {}
    if isinstance(app.state.lstm_bundle, dict):
        model = app.state.lstm_bundle.get("model")
        config = app.state.lstm_bundle.get("config", {}) or {}

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LSTM model object was not loaded",
        )

    top_k = int(config.get("top_k", 3))
    ratio_threshold = float(config.get("anomaly_ratio_threshold", 0.5))
    device = config.get("device", "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    windows = torch.tensor(windows_df["window"].tolist(), dtype=torch.long, device=device)
    targets = torch.tensor(windows_df["target"].tolist(), dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(windows)
        probs = torch.softmax(logits, dim=-1)
        k = min(top_k, probs.shape[1])
        topk = torch.topk(probs, k=k, dim=-1).indices
        correct = (topk == targets.unsqueeze(1)).any(dim=1)
        anomaly_ratio = (~correct).float().mean().item()

    prediction_lstm = "Anomaly" if anomaly_ratio > ratio_threshold else "Normal"
    probability = anomaly_ratio

    execution_time = perf_counter() - start_time
    token_count = len(str(request.feature).split())

    db_request = Logs(
        input_data=str(request.feature),
        result=prediction_lstm,
        probability=probability,
        execution_time=execution_time,
        token_count=token_count,
    )

    session.add(db_request)
    await session.commit()
    await session.refresh(db_request)

    response = PredictionResponse(
        prediction=prediction_lstm,
        probability=probability,
    )

    if tokens:
        response.access_token = tokens["access_token"]
        response.refresh_token = tokens["refresh_token"]
        response.token_type = tokens["token_type"]

    return response



@app.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: RefreshTokenRequest):
    """
    Обновление токенов с помощью refresh токена.

    Возвращает новую пару access и refresh токенов.
    """
    tokens = refresh_access_token(request.refresh_token)

    if tokens is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    return TokenResponse(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        token_type=tokens["token_type"],
    )


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    session: Session = Depends(get_session), current_user: str = Depends(require_auth)
):
    """
    Получение истории предсказаний.

    Требует аутентификации через Bearer token.
    """
    try:
        statement = select(Logs)
        results = await (session.exec(statement))
        results = results.all()
        serialized_result = {}
        for log in results:
            serialized_result[log.id] = DBItems(
                timestamp=str(log.timestamp),
                input_data=log.input_data,
                result=log.result,
                probability=log.probability,
                execution_time=log.execution_time,
                token_count=log.token_count,
            )
        return HistoryResponse(id=serialized_result)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"problem with db"
        )


@app.delete("/history")
async def delete_history(session: Session = Depends(get_session),
                   confirm_token: str = Header(default=None)):
    if confirm_token is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="bad request")
    if confirm_token != app.state.history_delete_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="forbidden")

    await session.execute(text("DELETE FROM logs"))
    await session.commit()
    return {"status": "ok"}

@app.get("/stats", response_model=StatsResponse)
async def get_stats(
    session: Session = Depends(get_session), current_user: str = Depends(require_auth)
):
    """
    Получение статистики по предсказаниям.

    Требует аутентификации через Bearer token.
    """
    try:
        sql_command = text("SELECT execution_time, token_count FROM logs")
        result = await session.execute(sql_command)
        result = result.fetchall()
        df_result = pd.DataFrame(result, columns=["execution_time", "token_count"])
        mean_execution_time = df_result["execution_time"].mean()
        q50 = df_result["execution_time"].quantile(0.5)
        q95 = df_result["execution_time"].quantile(0.95)
        q99 = df_result["execution_time"].quantile(0.99)
        mean_token_size = df_result["token_count"].mean()
        return StatsResponse(
            mean_execution_time=mean_execution_time,
            Q_50=q50,
            Q_95=q95,
            Q_99=q99,
            mean_token_size=mean_token_size,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=f"problem with stats"
        )


@app.get("/health")
async def health_check():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "if_model_loaded": app.state.if_model is not None,
        "if_threshold_loaded": app.state.if_threshold is not None,
        "lstm_model_loaded": app.state.lstm_bundle is not None
    }
