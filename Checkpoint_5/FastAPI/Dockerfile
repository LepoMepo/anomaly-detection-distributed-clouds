FROM python:3.11-slim

# Устанавливаем пользователя на раннем этапе
ARG USER_NAME=appuser
ARG USER_UID=1001
ARG USER_GID=1001

# Создаем не-root пользователя и группу
RUN groupadd --gid $USER_GID $USER_NAME && \
    useradd --uid $USER_UID --gid $USER_GID --shell /bin/bash --create-home $USER_NAME

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копирование файла зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY --chown=$USER_UID:$USER_GID . .

# Создание директории для базы данных с правами пользователя
RUN mkdir -p /app/db_data && chown $USER_UID:$USER_GID /app/db_data

USER $USER_NAME

# Открытие порта
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
