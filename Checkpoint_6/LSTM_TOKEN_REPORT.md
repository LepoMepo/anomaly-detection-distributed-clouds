# Отчет по Checkpoint 6 (Богданов Андрей)

## 1. Цель
Улучшить LSTM-подход из Checkpoint 5: вместо одного прогноза `T+1` для всего окна модель предсказывает следующий
event_id для каждой позиции внутри окна.

## 2. Отличие от baseline LSTM
Baseline:
```text
[e1, e2, ..., e10] -> e11
```

Many-to-many next-token:
```text
[e1, e2, ..., e10] -> [e2, e3, ..., e11]
```

Так одно окно дает `window_size` обучающих сигналов, а не один.

## 3. Реализованные файлы
- `FastAPI/model/SequenceTokenTransformer.py` - формирует `window` и `target_window`
- `FastAPI/model/lstm_token_model.py` - LSTM, возвращающая logits для каждой позиции окна
- `FastAPI/model/lstm_token_metrics.py` - NLL/top-k scoring и block-level метрики
- `scripts/prepare_hdfs_token_sequences.py` - подготовка датасета `hdfs_token_sequence_data.npz`
- `scripts/train_lstm_token.py` - обучение many-to-many LSTM
- `POST /forward/lstm-token` - инференс в FastAPI

## 4. Скоринг аномалий
Модель обучается через `CrossEntropyLoss`, то есть минимизирует NLL (`-log P(true_next_event)`) по всем позициям окна.
После обучения были сравнены несколько стратегий агрегации score на уровне блока:

```text
topk_all, topk_last, topk_last3, nll_mean, nll_p95, nll_max
```

Лучшей стратегией оказался `nll_max`:

```text
block_score = max(-log P(true_next_event))
```

Если `block_score >= threshold`, блок считается аномальным. Порог подбирается по F1 на validation.

Итог на test при threshold, подобранном на validation:
- `precision` = 0.960
- `recall` = 0.877
- `f1` = 0.917
- `fpr` = 0.0049
- `average precision` = 0.975

## 5. Подготовка данных
```powershell
python .\Checkpoint_6\scripts\prepare_hdfs_token_sequences.py `
  --log-path ".\Checkpoint_6\data\HDFS.log" `
  --label-path ".\Checkpoint_6\data\preprocessed\anomaly_label.csv" `
  --drain-state ".\Checkpoint_6\FastAPI\model\drain3_state_lstm_token.bin" `
  --drain-config ".\Checkpoint_6\FastAPI\model\drain3.ini" `
  --output-dir ".\Checkpoint_6\data\preprocessed\token_seq_out" `
  --window-size 10 `
  --stride 1 `
  --train-norm-ratio 0.8 `
  --val-norm-ratio 0.1 `
  --val-anom-ratio 0.3
```

После подготовки нужно перенести `sequence_token_transformer.joblib` в `Checkpoint_6/FastAPI/model/`.

## 6. Обучение
```powershell
python .\Checkpoint_6\scripts\train_lstm_token.py `
  --data ".\Checkpoint_6\data\preprocessed\token_seq_out\hdfs_token_sequence_data.npz" `
  --model-out ".\Checkpoint_6\FastAPI\model\lstm_token_model.pt" `
  --history-out ".\Checkpoint_6\experiments\lstm_token_model.history.json" `
  --epochs 5 `
  --batch-size 128 `
  --embedding-dim 32 `
  --hidden-size 64 `
  --num-layers 1 `
  --scoring nll_max `
  --device cuda
```

## 7. API
Новый endpoint:
```text
POST /forward/lstm-token
```

Требует минимум `window_size + 1` строк одного `block_id`, потому что для каждого окна нужны input-события и
следующие target-события.

Endpoint возвращает в поле `probability` значение `block_score` (`nll_max`), а не вероятность класса.
