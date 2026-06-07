import pandas as pd

from model.SequenceTransformer import SequenceTransformer


class SequenceTokenTransformer(SequenceTransformer):
    def transform(self, X):
        messages = X[self.message_col].astype(str)
        df = pd.DataFrame(index=X.index)

        df["date"] = messages.apply(self.get_date)
        df["time"] = messages.apply(self.get_time)
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format=self.timestamp_format,
            errors="coerce",
        )
        df["block_id"] = messages.apply(self.get_block_id)
        df["cluster_id"] = messages.apply(self.get_cluster_id)

        if self.drop_unknown_block:
            df = df[df["block_id"].notna()]

        df = df.dropna(subset=["timestamp"])
        df["event_id"] = df["cluster_id"].apply(
            lambda cid: self.event_to_id_.get(cid, self.event_to_id_["unknown"])
        )
        df = df.sort_values(["block_id", "timestamp"])

        sequences = []
        block_ids = []
        for block_id, group in df.groupby("block_id"):
            events = group["event_id"].tolist()
            if len(events) < self.min_sequence_len:
                continue
            sequences.append(events)
            block_ids.append(block_id)

        if not self.return_windows:
            return pd.DataFrame({"block_id": block_ids, "sequence": sequences})

        windows = []
        target_windows = []
        window_block_ids = []
        for block_id, seq in zip(block_ids, sequences):
            if len(seq) <= self.window_size:
                continue
            for i in range(0, len(seq) - self.window_size, self.stride):
                windows.append(seq[i : i + self.window_size])
                target_windows.append(seq[i + 1 : i + self.window_size + 1])
                window_block_ids.append(block_id)

        return pd.DataFrame(
            {
                "block_id": window_block_ids,
                "window": windows,
                "target_window": target_windows,
            }
        )
