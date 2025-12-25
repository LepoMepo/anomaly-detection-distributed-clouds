# Годовой проект HSE ИИ25 — Команда 33  
## Детектирование аномалий в данных распределённых облаков

## Краткое описание выполненного этапа (Checkpoint 4)

Вклад участников:
1. Ванюшин Павел - реализован базовый функционал (эндпоинты `/forward`, `/history`, `/stats`)
Обучение парсера данных drain и модели Isolation Forest. Реализация базы данных SQLite.
2. Иванов Артём - реализован механизм авторизации при помощи JWT. 
Упаковывание приложения в Docker контейнер. Реализация дополнительных эндпоинтов `/refresh`, `/health`.
3. Богданов Андрей - реализован эндпоинт `DELETE` `/history`. 
Добавлен хендлер, заменяющий стандартную ошибку 422, которую FastAPI возвращает при ошибке валидации, 
на 400 в соответствии с заданием.

**Описание и ссылки:**
Приложение на FastAPI с эндпоинтами `/forward`, `/history`, `/stats`, `/refresh`, `/health`, `/delete`. 
Версия python 3.11 (в ином случае модель не запустится).
Использует сырые логи для предсказания, один запрос должен в себя включать временной промежуток не более 10 секунд.
Пример json файла приведен в папке ([тут](https://github.com/LepoMepo/anomaly-detection-distributed-clouds/blob/main/Checkpoint_4/FastAPI/json_example.json)).
В качестве базы данных используется SQLite. Используется предобученный парсер drain.
([FastAPI](https://github.com/LepoMepo/anomaly-detection-distributed-clouds/tree/main/Checkpoint_4/FastAPI))

## Новые функции

### Docker
Приложение теперь упаковано в Docker контейнер для удобного развертывания.

### JWT Аутентификация
Добавлена двухтокенная система аутентификации:
- **Access Token** — короткоживущий токен (30 минут по умолчанию)
- **Refresh Token** — долгоживущий токен (7 дней по умолчанию)

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

### POST /forward — Предсказание аномалий
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

