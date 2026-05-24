from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def evaluate_top1_accuracy(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int = 1024,
) -> float:
    """
    Считает top-1 accuracy

    Args:
        model: LSTM-модель
        x: массив окон (N, window_size)
        y: массив целевых событий (N,)
        device: "cpu" или "cuda"
        batch_size: размер батча для инференса

    Returns:
        Доля окон, где argmax по логитам совпал с целевым событием
    """
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


def predict_topk_miss(
    model,
    x: np.ndarray,
    y: np.ndarray,
    device: str,
    top_k: int,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Возвращает массив miss по окнам (1 если target не попал в top-k)

    Args:
        model: LSTM-модель
        x: массив окон (N, window_size)
        y: массив целевых событий (N,)
        device: "cpu" или "cuda"
        top_k: размер top-k для проверки попадания
        batch_size: размер батча для инференса

    Returns:
        Массив miss (0/1) длины N
    """
    model.eval()
    if x.size == 0:
        return np.array([], dtype=np.int32)
    miss = np.zeros(len(x), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.long, device=device)
            yb = torch.tensor(y[i : i + batch_size], dtype=torch.long, device=device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            k = min(top_k, probs.shape[1])
            topk = torch.topk(probs, k=k, dim=-1).indices
            correct = (topk == yb.unsqueeze(1)).any(dim=1)
            miss[i : i + len(yb)] = (~correct).cpu().numpy().astype(np.int32)
    return miss


def block_anomaly_ratio_map(miss: np.ndarray, block_ids: np.ndarray) -> dict[str, float]:
    """
    Считает anomaly_ratio по каждому block_id

    anomaly_ratio = (число miss по блоку) / (число окон блока)

    Args:
        miss: массив miss (0/1) по окнам
        block_ids: массив block_id для каждого окна

    Returns:
        Словарь {block_id: anomaly_ratio}
    """
    block_ids = block_ids.astype(str)
    block_to_miss = {}
    block_to_total = {}
    for bid, m in zip(block_ids, miss):
        block_to_miss[bid] = block_to_miss.get(bid, 0) + int(m)
        block_to_total[bid] = block_to_total.get(bid, 0) + 1
    return {bid: block_to_miss[bid] / block_to_total[bid] for bid in block_to_total}


def block_anomaly_ratio(miss: np.ndarray, block_ids: np.ndarray) -> np.ndarray:
    """
    Возвращает только значения anomaly_ratio без идентификаторов блоков

    Args:
        miss: массив miss (0/1) по окнам
        block_ids: массив block_id для каждого окна

    Returns:
        Массив anomaly_ratio
    """

    ratios = block_anomaly_ratio_map(miss, block_ids)
    return np.array(list(ratios.values()), dtype=float)


def block_confusion_from_ratio_map(
    ratio_map: dict[str, float],
    label_map: dict[str, int],
    threshold: float,
) -> tuple[int, int, int, int]:
    """
    Считает TP/FP/TN/FN по готовой карте anomaly_ratio

    Args:
        ratio_map: словарь {block_id: anomaly_ratio}
        label_map: словарь {block_id: 0/1} с истинной меткой
        threshold: порог для anomaly_ratio

    Returns:
        (tp, fp, tn, fn)
    """
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


def block_level_metrics(
    miss: np.ndarray,
    block_ids: np.ndarray,
    labels: dict[str, int],
    threshold: float,
) -> tuple[float, float, float]:
    """
    Считает Precision/Recall/F1 на уровне блоков по порогу anomaly_ratio

    Логика:
    1) агрегируем miss по блокам
    2) считаем anomaly_ratio
    3) сравниваем с порогом threshold
    4) считаем P/R/F1 по block_id

    Args:
        miss: массив miss (0/1) по окнам
        block_ids: массив block_id для каждого окна
        labels: словарь {block_id: 0/1} с истинной меткой
        threshold: порог для anomaly_ratio

    Returns:
        (precision, recall, f1) на уровне блоков
    """
    ratios = block_anomaly_ratio_map(miss, block_ids)
    tp, fp, tn, fn = block_confusion_from_ratio_map(ratios, labels, threshold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def sweep_thresholds(
    ratio_map: dict[str, float],
    label_map: dict[str, int],
    thresholds: Iterable[float],
) -> list[tuple[float, float, float, float]]:
    """
    Считает Precision/Recall/F1 для набора порогов

    Args:
        ratio_map: словарь {block_id: anomaly_ratio}
        label_map: словарь {block_id: 0/1} с истинной меткой
        thresholds: набор порогов для перебора

    Returns:
        Список кортежей (threshold, precision, recall, f1)
    """
    results = []
    for thr in thresholds:
        tp, fp, tn, fn = block_confusion_from_ratio_map(ratio_map, label_map, thr)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        results.append((float(thr), precision, recall, f1))
    return results
