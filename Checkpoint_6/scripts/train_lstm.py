import argparse
import json
import sys

import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTAPI_DIR = ROOT / "FastAPI"
sys.path.insert(0, str(FASTAPI_DIR))

from model.lstm_model import LSTMNextEventModel
from model.lstm_metrics import (
    evaluate_top1_accuracy,
    predict_topk_miss,
    block_anomaly_ratio,
    block_level_metrics,
)


def _load_npz(path: Path):
    # Загружаем подготовленные окна и метки по блокам
    data = np.load(path, allow_pickle=True)
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        data["val_labels"],
        data["val_block_ids"],
        data["test_labels"],
        data["test_block_ids"],
    )


def main():
    # Парсер аргументов
    parser = argparse.ArgumentParser(description="Train LSTM for next-event prediction.")
    parser.add_argument("--data", required=True, help="Path to hdfs_sequence_data.npz")
    parser.add_argument("--model-out", required=True, help="Path to save lstm_model.pt")
    parser.add_argument("--history-out", default=None, help="Path to save training history json")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--anomaly-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Загружаем подготовленные данные (train/val/test)
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        val_labels,
        val_block_ids,
        test_labels,
        test_block_ids,
    ) = _load_npz(Path(args.data))
    if x_train.size == 0:
        raise ValueError("X_train is empty")

    # Размер словаря событий = max(event_id) + 1
    vocab_size = int(max(x_train.max(), y_train.max(), x_test.max(), y_test.max()) + 1)

    # Инициализация LSTM-модели
    model = LSTMNextEventModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Перенос модели на устройство
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    # Настройка оптимизации
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dataloader для обучения
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=args.batch_size,
        shuffle=True
    )

    # Карта block_id -> метка для val/test
    val_label_map = {str(bid): int(lbl) for bid, lbl in zip(val_block_ids, val_labels)}
    test_label_map = {str(bid): int(lbl) for bid, lbl in zip(test_block_ids, test_labels)}

    def _compute_loss(model, x_data, y_data):
        if x_data.size == 0:
            return 0.0
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for i in range(0, len(x_data), args.batch_size):
                xb = torch.tensor(x_data[i : i + args.batch_size], dtype=torch.long, device=device)
                yb = torch.tensor(y_data[i : i + args.batch_size], dtype=torch.long, device=device)
                logits = model(xb)
                loss_val = criterion(logits, yb)
                total_loss += loss_val.item() * yb.size(0)
                total_count += yb.size(0)
        return total_loss / total_count if total_count else 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": [],
        "val_f1": [],
        "val_ratio_mean": [],
        "threshold": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": []
    }

    # Основной цикл обучения
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Оценка на валидации (по окнам)
        val_acc = (
            evaluate_top1_accuracy(model, x_val, y_val, device, args.batch_size)
            if x_val.size
            else 0.0
        )
        val_miss = np.array([], dtype=np.int32)
        best_thr = args.anomaly_ratio_threshold
        if x_val.size and len(val_block_ids):
            # Miss по валидации и расчёт anomaly_ratio по блокам
            val_miss = predict_topk_miss(
                model,
                x_val,
                y_val,
                device,
                args.top_k,
                args.batch_size
            )
            # Подбор порога на валидации (максимум F1, если есть аномалии)
            ratios = block_anomaly_ratio(val_miss, val_block_ids)
            if ratios.size:
                if np.any(val_labels):
                    max_thr = float(np.quantile(ratios, 0.995))
                    thresholds = np.linspace(0.0, max(0.2, max_thr), 50)
                    best_f1 = -1.0
                    for thr in thresholds:
                        _, _, f1 = block_level_metrics(val_miss, val_block_ids, val_label_map, float(thr))
                        if f1 > best_f1:
                            best_f1 = f1
                            best_thr = float(thr)
            # Метрики по валидации
            val_ratio_mean = float(ratios.mean()) if ratios.size else 0.0
        else:
            val_ratio_mean = 0.0

        if x_val.size and np.any(val_labels):
            _, _, val_f1 = block_level_metrics(
                val_miss,
                val_block_ids,
                val_label_map,
                best_thr
            )
        else:
            val_f1 = 0.0

        # Оценка на тесте (accuracy по окнам + F1 по блокам)
        test_acc = evaluate_top1_accuracy(model, x_test, y_test, device, args.batch_size)
        # Miss по тесту
        test_miss = predict_topk_miss(
            model,
            x_test,
            y_test,
            device,
            args.top_k,
            args.batch_size
        )
        # Расчет F1 на тесте по блокам при найденном пороге
        t_precision, t_recall, t_f1 = block_level_metrics(test_miss, test_block_ids, test_label_map, best_thr)

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_loss = _compute_loss(model, x_val, y_val)

        history["train_loss"].append(float(avg_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["test_acc"].append(float(test_acc))
        history["val_f1"].append(float(val_f1))
        history["val_ratio_mean"].append(float(val_ratio_mean))
        history["threshold"].append(float(best_thr))
        history["test_precision"].append(float(t_precision))
        history["test_recall"].append(float(t_recall))
        history["test_f1"].append(float(t_f1))

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss: {avg_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"val_f1: {val_f1:.4f} | thr: {best_thr:.2f} | "
            f"test_F1: {t_f1:.4f} (P={t_precision:.4f}, R={t_recall:.4f})"
        )

    # Сохранение модели и конфигурации
    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "config": {
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "top_k": args.top_k,
                "anomaly_ratio_threshold": float(best_thr),
                "device": device
            }
        },
        out_path
    )

    if args.history_out:
        history_path = Path(args.history_out)
    else:
        history_path = out_path.with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
