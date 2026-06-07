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

## 5. Ablation на предыдущей one-step LSTM

Дополнительно те же стратегии подсчета anomaly score были применены к предыдущей LSTM из Checkpoint 5, которая
предсказывает только следующий event после окна:

```text
[e1, e2, ..., e10] -> e11
```

Цель эксперимента - отделить вклад новой many-to-many архитектуры от вклада нового способа scoring. Для всех стратегий
порог подбирался только на validation, после чего итоговые метрики считались на test.

Итог для `topk_miss`:
- `threshold` = 0.0909
- `precision` = 0.795
- `recall` = 0.781
- `f1` = 0.788
- `fpr` = 0.0269
- `average precision` = 0.787

Итог для `nll_mean`:
- `threshold` = 0.8501
- `precision` = 0.969
- `recall` = 0.872
- `f1` = 0.918
- `fpr` = 0.0038
- `average precision` = 0.982

Итог для `nll_p95`:
- `threshold` = 3.9213
- `precision` = 0.975
- `recall` = 0.884
- `f1` = 0.927
- `fpr` = 0.0030
- `average precision` = 0.975

Итог для `nll_max`:
- `threshold` = 6.3025
- `precision` = 0.976
- `recall` = 0.924
- `f1` = 0.950
- `fpr` = 0.0030
- `average precision` = 0.986

Результат показывает, что основной прирост качества связан не только с переходом к many-to-many next-token модели,
но и с заменой грубого `top-k miss` score на вероятностный NLL-based score. Предыдущая one-step LSTM тоже поддерживает
такой способ подсчета, потому что ее `CrossEntropyLoss` соответствует минимизации `-log P(true_next_event)`.

На текущем прогоне лучший практический вариант по test F1 - previous one-step LSTM с `nll_max` scoring:

```text
test_f1 = 0.950
average_precision = 0.986
```

Ma*ny-to-many модель оставлена как исследовательское улучшение и основа для дальнейшего перехода к MLM/LogBERT-подходам,
но сам по себе many-to-many objective на этом датасете не превзошел one-step LSTM при одинаковом NLL-based scoring.*

## 6. Подготовка данных
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

## 7. Обучение
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

## 8. API
Новый endpoint:
```text
POST /forward/lstm-token
```

Требует минимум `window_size + 1` строк одного `block_id`, потому что для каждого окна нужны input-события и
следующие target-события.

Endpoint возвращает в поле `probability` значение `block_score` (`nll_max`), а не вероятность класса.
