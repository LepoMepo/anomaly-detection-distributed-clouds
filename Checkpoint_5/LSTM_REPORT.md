# Отчет по Checkpoint 5 (Богданов Андрей)

## 1. Цель работы
Реализовать детектор аномалий в HDFS логах на основе LSTM.  
Модель должна обучаться на нормальных блоках, а аномалии выявляться по отклонениям
в последовательностях событий.

## 2. Что сделано
1. Подготовка последовательностей логов по `block_id` через Drain3
2. LSTM для next‑event prediction (обучение на нормальных блоках) + порог по F1 на валидации
3. Оценка качества по блокам через `anomaly_ratio` 
4. Инференс в FastAPI на `/forward/lstm`
5. Ноутбук с графиками для анализа качества

## 3. Данные
Источник: HDFS log (LogHub), разметка по `block_id`:
- `HDFS.log`
- `anomaly_label.csv`

Разметка дана на уровне блоков, поэтому последовательности формируются внутри `block_id`.

## 4. Предобработка
Используется Drain3 для шаблонизации логов:
- каждая строка преобразуется в `cluster_id`
- затем `cluster_id` кодируется в `event_id`

Из событий внутри блока формируются окна:
- `window_size = 10`
- `stride = 1`

Для каждого окна цель - следующее событие (next‑event prediction).

Блоки делятся на:
- **train**: только нормальные блоки
- **val**: нормальные + аномальные блоки 
- **test**: нормальные + аномальные блоки

## 4. Модель
LSTM решает задачу next‑event prediction

Архитектура: embedding → LSTM → linear → logits

Обучение: `CrossEntropyLoss` по прогнозу следующего события

Метрика по окнам: top‑1 accuracy

## 5. Скоринг аномалий
Для каждого блока считаем:
```
anomaly_ratio = (число окон, где target не попал в top‑k) / (общее число окон блока)
```

Если `anomaly_ratio >= threshold`, блок считается аномальным.

## 6. Подбор порога
Порог выбирается по максимуму F1 на валидации.

## 7. Оценка качества
На тесте считаются Precision/Recall/F1 на уровне блоков

Дополнительно в ноутбуке строятся:
- матрица ошибок и Precision/Recall/F1 (test)
- гистограммы `anomaly_ratio` (val/test)
- график зависимости FPR от порога (val)
- график зависимости Precision/Recall/F1 от порога (test)

## 8. Результаты
Полученные метрики на test (при threshold=0.09, подобранном по F! на val):
- `precision` = 0.820 (большая часть срабатываний действительно аномалии)
- `recall` = 0.740 (около 26% аномалий пропущено)
- `f1` = 0.778 (хорошее качество с упором на точность)
- `fpr` = 0.022 (почти нет ложных срабатываний на норме)

Использованные гиперпараметры:
- Последовательности:
  - `window_size=10`, `stride=1`
- Модель:
  - `embedding_dim=32`, `hidden_size=64`
  - `num_layers=1`, `dropout=0.0`
  - `batch_size=128`, `epochs=10`
  - `top_k=3`

## 9. Артефакты
Для сервиса нужны:
- `lstm_model.pt` (веса LSTM и сохраненные гиперпараметры)
- `sequence_transformer.joblib` (Drain3‑трансформер + параметры окон/словари событий)
- `drain3_state_lstm.bin` (состояние шаблонов Drain3, на которых обучен transformer))
- `drain3.ini` (конфигурация Drain3)

## 10. Полный прогон (подготовка → обучение)
### 1) Подготовка данных
`Checkpoint_5\data\` должна содержать:
- `HDFS.log` - сырые логи
- `preprocessed\anomaly_label.csv` - разметку по блокам
```powershell
python .\Checkpoint_5\scripts\prepare_hdfs_sequences.py `
  --log-path ".\Checkpoint_5\data\HDFS.log" `
  --label-path ".\Checkpoint_5\data\preprocessed\anomaly_label.csv" `
  --drain-state ".\Checkpoint_5\FastAPI\model\drain3_state_lstm.bin" `
  --drain-config ".\Checkpoint_5\FastAPI\model\drain3.ini" `
  --output-dir ".\Checkpoint_5\data\preprocessed\seq_out" `
  --window-size 10 `
  --stride 1 `
  --train-norm-ratio 0.8 `
  --val-norm-ratio 0.1 `
  --val-anom-ratio 0.3
```
Выход:
- `Checkpoint_5/FastAPI/model/drain3_state_lstm.bin`
- `seq_out/hdfs_sequence_data.npz` 
- `seq_out/sequence_transformer.joblib` (перенести в `Checkpoint_5/FastAPI/model/` для инференса)
- `seq_out/hdfs_sequence_meta.json`

### 2) Обучение LSTM
```powershell
python .\Checkpoint_5\scripts\train_lstm.py `
  --data ".\Checkpoint_5\data\preprocessed\seq_out\hdfs_sequence_data.npz" `
  --model-out ".\Checkpoint_5\FastAPI\model\lstm_model.pt" `
  --history-out ".\Checkpoint_5\experiments\lstm_model.history.json" `
  --epochs 5 `
  --batch-size 128 `
  --embedding-dim 32 `
  --hidden-size 64 `
  --num-layers 1 `
  --top-k 3 `
  --device cuda
```
