# Годовой проект HSE ИИ25 — Команда 33  
## Детектирование аномалий в данных распределённых облаков

## Краткое описание выполненного этапа (Checkpoint 5)

Вклад участников:
1. Ванюшин Павел - 

2. Иванов Артём -

3. Богданов Андрей - реализован LSTM-пайплайн: подготовка последовательностей по block_id 
(`prepare_hdfs_sequences.py`, `SequenceTransformer`), обучение модели (`lstm_model.py`, `train_lstm.py`), 
метрики и подбор порога (`lstm_metrics.py`), ноутбук с анализом метрик (`lstm_eval.ipynb`), 
интеграция с FastAPI (`/forward/lstm`) и отчет (`LSTM_REPORT.md`).

## Новые функции

### API эндпоинты
POST /forward/lstm — Предсказание аномалий LSTM моделью (требуется минимум window_size + 1 строк одного и того же блока)

POST /forward и /forward/if — предыдущая IF модель.

## Запуск с Docker

### Быстрый старт
```bash
cd FastAPI
docker-compose up --build
```

### Или без docker-compose
```bash
cd FastAPI
docker build -t anomaly-detection-api .
docker run -p 8000:8000 --env-file .env anomaly-detection-api
```

После запуска API доступен по адресу: http://localhost:8000

Swagger документация: http://localhost:8000/docs

## API Эндпоинты

### POST /forward и /forward/if — Предсказание аномалий моделью IF
### POST /forward/lstm — Предсказание аномалий моделью LSTM
Аутентификация может быть выполнена двумя способами:

**Вариант 1: Логин/пароль в теле запроса (получение токенов)**
```json
{
  "feature_name": "original_message",
  "feature": ["081109 203518 143 INFO dfs.DataNode$DataXceiver: ..."],
  "username": "admin",
  "password": "admin123"
}
```

Ответ включает токены:
```json
{
  "prediction": "Normal",
  "probability": 0.123,
  "access_token": "eyJ...",
  "refresh_token": "eyJ...",
  "token_type": "bearer"
}
```

**Вариант 2: Bearer токен в заголовке**
```bash
curl -X POST http://localhost:8000/forward \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"feature_name": "original_message", "feature": [...]}'
```

### POST /refresh — Обновление токенов
```json
{
  "refresh_token": "eyJ..."
}
```

### GET /history — История предсказаний
Требует Bearer токен в заголовке.

### DELETE /history - Удаление истории обращений из базы данных
Требует confirm-token в заголовке.

### GET /stats — Статистика
Требует Bearer токен в заголовке.

### GET /health — Проверка состояния сервиса
Публичный эндпоинт.

## Демо учетные данные
- **Username:** admin
- **Password:** admin123

## Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|----------------------|
| SECRET_KEY | Секретный ключ для JWT | your_token           |
| ALGORITHM | Алгоритм шифрования | HS256                |
| ACCESS_TOKEN_EXPIRE_MINUTES | Время жизни access токена | 30                   |
| REFRESH_TOKEN_EXPIRE_DAYS | Время жизни refresh токена | 7                    |

## Участники:
- <Участник 1, Иванов Артём> — @Vanarti, Vanarty
- <Участник 2, Богданов Андрей> — @wanna_sleeeep, andrewb-codes
- <Участник 3, Кузнецов Виталий> — @pismith, Vitaly
- <Участник 4, Ванюшин Павел> — @LepoMepo, LepoMepo

