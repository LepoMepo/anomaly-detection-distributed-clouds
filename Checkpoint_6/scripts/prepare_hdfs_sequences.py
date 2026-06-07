import argparse
import json
import re
import sys

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTAPI_DIR = ROOT / "FastAPI"
sys.path.insert(0, str(FASTAPI_DIR))

from model.SequenceTransformer import SequenceTransformer


def _read_labels(label_path: Path) -> pd.DataFrame:
    df = pd.read_csv(label_path, usecols=["BlockId", "Label"])
    df = df.rename(columns={"BlockId": "block_id", "Label": "label"})
    df["is_anomaly"] = df["label"].astype(str).str.strip().str.lower().eq("anomaly").astype(int)
    return df


def _split_blocks(
    labels_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    val_anom_ratio: float,
    seed: int
):
    normal_blocks = (
        labels_df[labels_df["is_anomaly"] == 0]["block_id"]
        .astype(str)
        .to_numpy()
    )
    anomaly_blocks = (
        labels_df[labels_df["is_anomaly"] == 1]["block_id"]
        .astype(str)
        .to_numpy()
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(normal_blocks)
    rng.shuffle(anomaly_blocks)

    train_size = int(len(normal_blocks) * train_ratio)
    val_size = int(len(normal_blocks) * val_ratio)
    val_start = train_size
    val_end = train_size + val_size

    train_blocks = set(normal_blocks[:train_size])
    val_blocks = set(normal_blocks[val_start:val_end])
    test_blocks = set(normal_blocks[val_end:]) | set(anomaly_blocks)

    if anomaly_blocks.size:
        val_anom_size = int(len(anomaly_blocks) * val_anom_ratio)
        val_anom_blocks = set(anomaly_blocks[:val_anom_size])
        test_anom_blocks = set(anomaly_blocks[val_anom_size:])
        val_blocks |= val_anom_blocks
        test_blocks = (set(normal_blocks[val_end:]) | test_anom_blocks)

    return train_blocks, val_blocks, test_blocks


def _split_log_lines(
    log_path: Path,
    train_blocks: set,
    val_blocks: set,
    test_blocks: set,
    block_id_regex: str,
    max_lines: int | None
):
    block_re = re.compile(block_id_regex)
    train_lines = []
    val_lines = []
    test_lines = []
    seen = 0

    with log_path.open("r", errors="ignore") as f:
        for line in f:
            if max_lines is not None and seen >= max_lines:
                break
            seen += 1
            match = block_re.search(line)
            if not match:
                continue
            block_id = match.group(1)
            if block_id in train_blocks:
                train_lines.append(line.rstrip("\n"))
            elif block_id in val_blocks:
                val_lines.append(line.rstrip("\n"))
            elif block_id in test_blocks:
                test_lines.append(line.rstrip("\n"))

    return train_lines, val_lines, test_lines


def _windows_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df["window"].tolist(), dtype=np.int64)


def _targets_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df["target"].tolist(), dtype=np.int64)


def _labels_for_windows(df: pd.DataFrame, label_map: dict) -> np.ndarray:
    labels = df["block_id"].map(label_map).fillna(0).astype(int)
    return labels.to_numpy()


def _save_outputs(
    output_dir: Path,
    transformer: SequenceTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict,
    train_blocks: set,
    val_blocks: set,
    test_blocks: set
):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(transformer, output_dir / "sequence_transformer.joblib")

    x_train = _windows_to_array(train_df)
    y_train = _targets_to_array(train_df)
    x_val = _windows_to_array(val_df)
    y_val = _targets_to_array(val_df)
    x_test = _windows_to_array(test_df)
    y_test = _targets_to_array(test_df)
    val_labels = _labels_for_windows(val_df, label_map)
    test_labels = _labels_for_windows(test_df, label_map)
    val_block_ids = val_df["block_id"].astype(str).to_numpy()
    test_block_ids = test_df["block_id"].astype(str).to_numpy()

    np.savez_compressed(
        output_dir / "hdfs_sequence_data.npz",
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        val_labels=val_labels,
        test_labels=test_labels,
        val_block_ids=val_block_ids,
        test_block_ids=test_block_ids
    )

    anomaly_block_set = {bid for bid, is_anom in label_map.items() if is_anom == 1}
    val_anom_blocks = val_blocks & anomaly_block_set
    test_anom_blocks = test_blocks & anomaly_block_set

    meta = {
        "train_windows": int(x_train.shape[0]),
        "val_windows": int(x_val.shape[0]),
        "test_windows": int(x_test.shape[0]),
        "window_size": int(x_train.shape[1]) if x_train.size else 0,
        "event_vocab_size": len(transformer.template_list_ or []),
        "unknown_event_id": transformer.event_to_id_.get("unknown", -1) if transformer.event_to_id_ else -1,
        "train_blocks": len(train_blocks),
        "val_blocks": len(val_blocks),
        "test_blocks": len(test_blocks),
        "val_anomaly_blocks": len(val_anom_blocks),
        "test_anomaly_blocks": len(test_anom_blocks)
    }
    with (output_dir / "hdfs_sequence_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Prepare HDFS sequences for LSTM training.")

    parser.add_argument("--log-path", required=True, help="Path to HDFS.log")
    parser.add_argument("--label-path", required=True, help="Path to anomaly_label.csv")
    parser.add_argument("--drain-state", required=True, help="Path to drain3_state.bin (will be created/updated)")
    parser.add_argument("--drain-config", required=True, help="Path to drain3.ini")
    parser.add_argument("--output-dir", required=True, help="Directory to save transformer and numpy HDFS_v1")
    parser.add_argument("--train-norm-ratio", type=float, default=0.8, help="Share of normal blocks for train")
    parser.add_argument("--val-norm-ratio", type=float, default=0.1, help="Share of normal blocks for val (rest goes to test)")
    parser.add_argument("--val-anom-ratio", type=float, default=0.3, help="Share of anomaly blocks in val (rest goes to test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--block-id-regex", default=r"(blk_[A-Za-z0-9_-]+)", help="Regex to extract block_id from log content")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    label_path = Path(args.label_path)
    output_dir = Path(args.output_dir)

    labels_df = _read_labels(label_path)
    label_map = dict(zip(labels_df["block_id"], labels_df["is_anomaly"]))

    if args.train_norm_ratio + args.val_norm_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    if not (0.0 <= args.val_anom_ratio <= 1.0):
        raise ValueError("val_anom_ratio must be between 0.0 and 1.0")

    train_blocks, val_blocks, test_blocks = _split_blocks(labels_df, args.train_norm_ratio, args.val_norm_ratio, args.val_anom_ratio, args.seed)

    train_lines, val_lines, test_lines = _split_log_lines(
        log_path,
        train_blocks,
        val_blocks,
        test_blocks,
        args.block_id_regex,
        args.max_lines
    )

    train_df = pd.DataFrame({"original_message": train_lines})
    val_df = pd.DataFrame({"original_message": val_lines})
    test_df = pd.DataFrame({"original_message": test_lines})

    transformer = SequenceTransformer(
        drain_state=str(Path(args.drain_state).resolve()),
        drain_config=str(Path(args.drain_config).resolve()),
        window_size=args.window_size,
        stride=args.stride,
        update_drain_on_fit=True
    )
    transformer.fit(train_df)

    train_windows = transformer.transform(train_df)
    val_windows = transformer.transform(val_df)
    test_windows = transformer.transform(test_df)

    _save_outputs(
        output_dir,
        transformer,
        train_windows,
        val_windows,
        test_windows,
        label_map,
        train_blocks,
        val_blocks,
        test_blocks
    )


if __name__ == "__main__":
    main()
