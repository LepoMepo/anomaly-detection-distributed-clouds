import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FASTAPI_DIR = ROOT / "FastAPI"
sys.path.insert(0, str(FASTAPI_DIR))

from model.SequenceTokenTransformer import SequenceTokenTransformer
from prepare_hdfs_sequences import _labels_for_windows, _read_labels, _split_blocks, _split_log_lines


def _windows_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df["window"].tolist(), dtype=np.int64)


def _target_windows_to_array(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(df["target_window"].tolist(), dtype=np.int64)


def _save_outputs(
    output_dir: Path,
    transformer: SequenceTokenTransformer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_map: dict,
    train_blocks: set,
    val_blocks: set,
    test_blocks: set,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(transformer, output_dir / "sequence_token_transformer.joblib")

    x_train = _windows_to_array(train_df)
    y_train = _target_windows_to_array(train_df)
    x_val = _windows_to_array(val_df)
    y_val = _target_windows_to_array(val_df)
    x_test = _windows_to_array(test_df)
    y_test = _target_windows_to_array(test_df)
    val_labels = _labels_for_windows(val_df, label_map)
    test_labels = _labels_for_windows(test_df, label_map)
    val_block_ids = val_df["block_id"].astype(str).to_numpy()
    test_block_ids = test_df["block_id"].astype(str).to_numpy()

    np.savez_compressed(
        output_dir / "hdfs_token_sequence_data.npz",
        X_train=x_train,
        Y_train=y_train,
        X_val=x_val,
        Y_val=y_val,
        X_test=x_test,
        Y_test=y_test,
        val_labels=val_labels,
        test_labels=test_labels,
        val_block_ids=val_block_ids,
        test_block_ids=test_block_ids,
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
        "unknown_event_id": transformer.event_to_id_.get("unknown", -1)
        if transformer.event_to_id_
        else -1,
        "train_blocks": len(train_blocks),
        "val_blocks": len(val_blocks),
        "test_blocks": len(test_blocks),
        "val_anomaly_blocks": len(val_anom_blocks),
        "test_anomaly_blocks": len(test_anom_blocks),
        "target": "next token at every position in the window",
    }
    with (output_dir / "hdfs_token_sequence_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HDFS windows for many-to-many LSTM next-token training."
    )
    parser.add_argument("--log-path", required=True, help="Path to HDFS.log")
    parser.add_argument("--label-path", required=True, help="Path to anomaly_label.csv")
    parser.add_argument("--drain-state", required=True, help="Path to drain3_state.bin")
    parser.add_argument("--drain-config", required=True, help="Path to drain3.ini")
    parser.add_argument("--output-dir", required=True, help="Directory to save token data")
    parser.add_argument("--train-norm-ratio", type=float, default=0.8)
    parser.add_argument("--val-norm-ratio", type=float, default=0.1)
    parser.add_argument("--val-anom-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--block-id-regex", default=r"(blk_[A-Za-z0-9_-]+)")
    args = parser.parse_args()

    labels_df = _read_labels(Path(args.label_path))
    label_map = dict(zip(labels_df["block_id"], labels_df["is_anomaly"]))

    if args.train_norm_ratio + args.val_norm_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    if not (0.0 <= args.val_anom_ratio <= 1.0):
        raise ValueError("val_anom_ratio must be between 0.0 and 1.0")

    train_blocks, val_blocks, test_blocks = _split_blocks(
        labels_df,
        args.train_norm_ratio,
        args.val_norm_ratio,
        args.val_anom_ratio,
        args.seed,
    )

    train_lines, val_lines, test_lines = _split_log_lines(
        Path(args.log_path),
        train_blocks,
        val_blocks,
        test_blocks,
        args.block_id_regex,
        args.max_lines,
    )

    train_raw = pd.DataFrame({"original_message": train_lines})
    val_raw = pd.DataFrame({"original_message": val_lines})
    test_raw = pd.DataFrame({"original_message": test_lines})

    transformer = SequenceTokenTransformer(
        drain_state=str(Path(args.drain_state).resolve()),
        drain_config=str(Path(args.drain_config).resolve()),
        window_size=args.window_size,
        stride=args.stride,
        update_drain_on_fit=True,
    )
    transformer.fit(train_raw)

    train_windows = transformer.transform(train_raw)
    val_windows = transformer.transform(val_raw)
    test_windows = transformer.transform(test_raw)

    _save_outputs(
        Path(args.output_dir),
        transformer,
        train_windows,
        val_windows,
        test_windows,
        label_map,
        train_blocks,
        val_blocks,
        test_blocks,
    )


if __name__ == "__main__":
    main()
