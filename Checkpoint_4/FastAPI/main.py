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

from settings.settings import MODEL_PATH, THRESHOLD_PATH, HISTORY_DELETE_TOKEN
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    import __main__

    __main__.LogTransformer = LogTransformer

    try:
        # with open(MODEL_PATH, "rb") as f:
        #     app.state.model = joblib.load(f)
        # with open(THRESHOLD_PATH, "rb") as f:
        #     app.state.threshold = joblib.load(f)

        # TO DO временный костыль для загрузки модели
        app.state.model = load_model_safely(MODEL_PATH)
        app.state.threshold = load_model_safely(THRESHOLD_PATH)

    except Exception as e:
        app.state.model = None
        app.state.threshold = None
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
    """
    Предсказание аномалий в логах.

    Аутентификация может быть выполнена двумя способами:
    1. Передать username и password в теле запроса (вернёт токены в ответе)
    2. Передать Bearer token в заголовке Authorization
    """
    start_time = perf_counter()

    tokens = None

    # Если нет токена в заголовке, проверяем логин/пароль в теле запроса
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

    if app.state.model is None or app.state.threshold is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model was not loaded",
        )

    if request.feature_name != "original_message":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="bad request"
        )

    try:
        df = pd.DataFrame({request.feature_name: request.feature})
        prediction = app.state.model["model"].decision_function(
            app.state.model["transformer"].transform(df)
        )[0]
        prediction_if = (prediction <= app.state.threshold).astype(int)
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
    if confirm_token != HISTORY_DELETE_TOKEN:
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
        "model_loaded": app.state.model is not None,
        "threshold_loaded": app.state.threshold is not None,
    }
