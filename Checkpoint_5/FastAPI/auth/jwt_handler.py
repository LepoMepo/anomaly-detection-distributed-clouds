from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from settings.settings import Settings

settings = Settings()

# Настройки JWT
SECRET_KEY = settings.jwt_secret_key.get_secret_value()
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = float(settings.jwt_access_token_expire_minutes)
REFRESH_TOKEN_EXPIRE_DAYS = float(settings.jwt_refresh_token_expire_days)

# Хеширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Демо пользователь в рамках проекта (в реальном приложении в проде будет храниться в БД)
DEMO_USERS = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверка пароля"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Хеширование пароля"""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Аутентификация пользователя"""
    user = DEMO_USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание access токена"""
    to_encode = data.copy()
    # Если передан expires_delta, используем его, иначе используем ACCESS_TOKEN_EXPIRE_MINUTES
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # Добавляем expire и type в токен
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создание refresh токена"""
    to_encode = data.copy()
    # Если передан expires_delta, используем его, иначе используем REFRESH_TOKEN_EXPIRE_DAYS
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    # Добавляем expire и type в токен
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_tokens(username: str) -> dict:
    """Создание пары токенов (access и refresh)"""
    access_token = create_access_token(data={"sub": username})
    refresh_token = create_refresh_token(data={"sub": username})
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """Верификация токена и извлечение username"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type_from_payload: str = payload.get("type")

        if username is None:
            return None
        if token_type_from_payload != token_type:
            return None

        return username
    except JWTError:
        return None


def refresh_access_token(refresh_token: str) -> Optional[dict]:
    """Обновление access токена с помощью refresh токена"""
    username = verify_token(refresh_token, token_type="refresh")
    if username is None:
        return None

    # Проверяем, что пользователь существует
    if username not in DEMO_USERS:
        return None

    # Создаем новую пару токенов
    return create_tokens(username)
