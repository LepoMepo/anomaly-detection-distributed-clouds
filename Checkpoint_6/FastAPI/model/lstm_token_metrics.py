from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def evaluate_token_top1_accuracy(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> float:
    model.eval()
    if x.size == 0:
        return 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.long, device=device)
            yb = torch.tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total if total else 0.0


def predict_token_topk_miss(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    top_k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    if x.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    miss_batches = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.long, device=device)
            yb = torch.tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            k = min(top_k, probs.shape[-1])
            topk = torch.topk(probs, k=k, dim=-1).indices
            correct = (topk == yb.unsqueeze(-1)).any(dim=-1)
            miss_batches.append((~correct).cpu().numpy().astype(np.int32))
    return np.concatenate(miss_batches, axis=0)


def predict_token_nll(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    if x.size == 0:
        return np.empty((0, 0), dtype=float)
    nll_batches = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.long, device=device)
            yb = torch.tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(dim=-1, index=yb.unsqueeze(-1)).squeeze(-1)
            nll_batches.append(nll.cpu().numpy())
    return np.concatenate(nll_batches, axis=0)


def block_token_score_map(
    scores: np.ndarray,
    block_ids: np.ndarray,
    aggregation: str = "max",
) -> dict[str, float]:
    block_ids = block_ids.astype(str)
    block_to_values = {}
    for bid, score_row in zip(block_ids, scores):
        block_to_values.setdefault(bid, []).append(np.asarray(score_row, dtype=float))

    result = {}
    for bid, rows in block_to_values.items():
        values = np.concatenate(rows)
        if aggregation == "max":
            result[bid] = float(values.max())
        elif aggregation == "mean":
            result[bid] = float(values.mean())
        elif aggregation == "p95":
            result[bid] = float(np.quantile(values, 0.95))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
    return result


def block_token_anomaly_ratio_map(
    miss: np.ndarray,
    block_ids: np.ndarray,
) -> dict[str, float]:
    block_ids = block_ids.astype(str)
    block_to_miss = {}
    block_to_total = {}
    for bid, miss_row in zip(block_ids, miss):
        block_to_miss[bid] = block_to_miss.get(bid, 0) + int(miss_row.sum())
        block_to_total[bid] = block_to_total.get(bid, 0) + int(miss_row.size)
    return {bid: block_to_miss[bid] / block_to_total[bid] for bid in block_to_total}


def block_token_anomaly_ratio(miss: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    ratios = block_token_anomaly_ratio_map(miss, block_ids)
    return np.array(list(ratios.values()), dtype=float)


def block_confusion_from_ratio_map(
    ratio_map: dict[str, float],
    label_map: dict[str, int],
    threshold: float,
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for bid, ratio in ratio_map.items():
        pred = 1 if ratio >= threshold else 0
        true = int(label_map.get(bid, 0))
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def block_level_token_metrics(
    miss: np.ndarray,
    block_ids: np.ndarray,
    labels: dict[str, int],
    threshold: float,
) -> tuple[float, float, float]:
    ratios = block_token_anomaly_ratio_map(miss, block_ids)
    tp, fp, tn, fn = block_confusion_from_ratio_map(ratios, labels, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def sweep_token_thresholds(
    ratio_map: dict[str, float],
    label_map: dict[str, int],
    thresholds: Iterable[float],
) -> list[tuple[float, float, float, float]]:
    results = []
    for thr in thresholds:
        tp, fp, tn, fn = block_confusion_from_ratio_map(ratio_map, label_map, thr)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        results.append((float(thr), precision, recall, f1))
    return results
