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

from model.lstm_token_model import LSTMNextTokenModel
from model.lstm_token_metrics import (
    block_confusion_from_ratio_map,
    block_token_score_map,
    evaluate_token_top1_accuracy,
    predict_token_nll,
)


def _load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return (
        data["X_train"],
        data["Y_train"],
        data["X_val"],
        data["Y_val"],
        data["X_test"],
        data["Y_test"],
        data["val_labels"],
        data["val_block_ids"],
        data["test_labels"],
        data["test_block_ids"],
    )


def _max_event_id(*arrays: np.ndarray) -> int:
    max_values = [int(arr.max()) for arr in arrays if arr.size]
    if not max_values:
        raise ValueError("No event ids found")
    return max(max_values)


def main():
    parser = argparse.ArgumentParser(
        description="Train many-to-many LSTM for next-token prediction."
    )
    parser.add_argument("--data", required=True, help="Path to hdfs_token_sequence_data.npz")
    parser.add_argument("--model-out", required=True, help="Path to save lstm_token_model.pt")
    parser.add_argument("--history-out", default=None)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--anomaly-score-threshold", type=float, default=1.0)
    parser.add_argument("--anomaly-ratio-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scoring", default="nll_max", choices=["nll_max"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.anomaly_ratio_threshold is not None:
        args.anomaly_score_threshold = args.anomaly_ratio_threshold

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

    vocab_size = _max_event_id(x_train, y_train, x_val, y_val, x_test, y_test) + 1

    model = LSTMNextTokenModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.long),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_label_map = {str(bid): int(lbl) for bid, lbl in zip(val_block_ids, val_labels)}
    test_label_map = {str(bid): int(lbl) for bid, lbl in zip(test_block_ids, test_labels)}

    def _compute_loss(x_data, y_data):
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
                loss_val = criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
                total_loss += loss_val.item() * yb.numel()
                total_count += yb.numel()
        return total_loss / total_count if total_count else 0.0

    history = {
        "scoring": args.scoring,
        "train_loss": [],
        "val_loss": [],
        "val_token_acc": [],
        "test_token_acc": [],
        "val_f1": [],
        "val_score_mean": [],
        "threshold": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    def _precision_recall_f1(ratio_map, label_map, threshold):
        tp, fp, tn, fn = block_confusion_from_ratio_map(ratio_map, label_map, threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return precision, recall, f1

    best_thr = args.anomaly_score_threshold
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_acc = (
            evaluate_token_top1_accuracy(model, x_val, y_val, device, args.batch_size)
            if x_val.size
            else 0.0
        )
        val_score_map = {}
        val_score_mean = 0.0
        if x_val.size and len(val_block_ids):
            val_nll = predict_token_nll(
                model,
                x_val,
                y_val,
                device,
                args.batch_size,
            )
            val_score_map = block_token_score_map(val_nll, val_block_ids, aggregation="max")
            scores = np.array(list(val_score_map.values()), dtype=float)
            val_score_mean = float(scores.mean()) if scores.size else 0.0
            if scores.size and np.any(val_labels):
                max_thr = float(np.quantile(scores, 0.995))
                thresholds = np.linspace(0.0, max(max_thr, float(scores.max()), 1e-9), 100)
                best_f1 = -1.0
                for thr in thresholds:
                    _, _, f1 = _precision_recall_f1(val_score_map, val_label_map, float(thr))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thr = float(thr)

        if x_val.size and np.any(val_labels):
            _, _, val_f1 = _precision_recall_f1(val_score_map, val_label_map, best_thr)
        else:
            val_f1 = 0.0

        test_acc = evaluate_token_top1_accuracy(model, x_test, y_test, device, args.batch_size)
        test_nll = predict_token_nll(
            model,
            x_test,
            y_test,
            device,
            args.batch_size,
        )
        test_score_map = block_token_score_map(test_nll, test_block_ids, aggregation="max")
        t_precision, t_recall, t_f1 = _precision_recall_f1(test_score_map, test_label_map, best_thr)

        avg_loss = epoch_loss / max(1, len(train_loader))
        val_loss = _compute_loss(x_val, y_val)

        history["train_loss"].append(float(avg_loss))
        history["val_loss"].append(float(val_loss))
        history["val_token_acc"].append(float(val_acc))
        history["test_token_acc"].append(float(test_acc))
        history["val_f1"].append(float(val_f1))
        history["val_score_mean"].append(float(val_score_mean))
        history["threshold"].append(float(best_thr))
        history["test_precision"].append(float(t_precision))
        history["test_recall"].append(float(t_recall))
        history["test_f1"].append(float(t_f1))

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss: {avg_loss:.4f} | val_loss: {val_loss:.4f} | "
            f"val_token_acc: {val_acc:.4f} | val_f1: {val_f1:.4f} | "
            f"scoring: {args.scoring} | thr: {best_thr:.2f} | test_F1: {t_f1:.4f} "
            f"(P={t_precision:.4f}, R={t_recall:.4f})"
        )

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
                "scoring": args.scoring,
                "anomaly_score_threshold": float(best_thr),
                "anomaly_ratio_threshold": float(best_thr),
                "device": device,
                "objective": "many_to_many_next_token",
            },
        },
        out_path,
    )

    history_path = Path(args.history_out) if args.history_out else out_path.with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
