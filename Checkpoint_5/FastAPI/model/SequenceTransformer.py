from sklearn.base import BaseEstimator, TransformerMixin
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
import re
import pandas as pd


class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        drain_state,
        drain_config,
        message_col="original_message",
        timestamp_format="%y%m%d %H%M%S",
        block_id_regex=r"(blk_[A-Za-z0-9_-]+)",
        window_size=10,
        stride=1,
        min_sequence_len=2,
        return_windows=True,
        return_next_event=True,
        update_drain_on_fit=True,
        drop_unknown_block=True,
        unknown_block_token="unknown_block",
    ):
        self.drain_state = drain_state
        self.drain_config = drain_config
        self.message_col = message_col
        self.timestamp_format = timestamp_format
        self.block_id_regex = block_id_regex
        self.window_size = window_size
        self.stride = stride
        self.min_sequence_len = min_sequence_len
        self.return_windows = return_windows
        self.return_next_event = return_next_event
        self.update_drain_on_fit = update_drain_on_fit
        self.drop_unknown_block = drop_unknown_block
        self.unknown_block_token = unknown_block_token
        persistance = FilePersistence(self.drain_state)
        config = TemplateMinerConfig()
        config.load(self.drain_config)
        self.template_miner = TemplateMiner(persistence_handler=persistance, config=config)
        self.pattern = re.compile(
            r'^(?P<Date>.+?)\s+(?P<Time>.+?)\s+(?P<Pid>.+?)\s+(?P<Level>.+?)\s+(?P<Component>.+?):\s+(?P<Content>.+?)$',
            re.IGNORECASE,
        )
        self.block_id_re = re.compile(self.block_id_regex)
        self.template_list_ = None
        self.event_to_id_ = None

    def fit(self, X, y=None):
        if self.update_drain_on_fit:
            for row in X[self.message_col]:
                content = self.get_content(row)
                self.template_miner.add_log_message(content)

        self.template_list_ = [cluster.cluster_id for cluster in self.template_miner.drain.clusters]
        self.template_list_.append("unknown")
        self.event_to_id_ = {cid: idx for idx, cid in enumerate(self.template_list_)}
        return self

    def transform(self, X):
        messages = X[self.message_col].astype(str)
        df = pd.DataFrame(index=X.index)

        # Парсим дату/время и собираем timestamp
        df["date"] = messages.apply(self.get_date)
        df["time"] = messages.apply(self.get_time)
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            format=self.timestamp_format,
            errors="coerce",
        )
        # Извлекаем block_id и сопоставляем каждой строке cluster_id Drain3
        df["block_id"] = messages.apply(self.get_block_id)
        df["cluster_id"] = messages.apply(self.get_cluster_id)

        # Если block_id не найден - отбрасываем
        if self.drop_unknown_block:
            df = df[df["block_id"].notna()]

        # Удаляем строки с некорректным timestamp
        df = df.dropna(subset=["timestamp"])

        # Преобразуем cluster_id в числовой event_id (с fallback на unknown)
        df["event_id"] = df["cluster_id"].apply(lambda cid: self.event_to_id_.get(cid, self.event_to_id_["unknown"]))

        # Сортируем внутри блоков по времени
        df = df.sort_values(["block_id", "timestamp"])

        # Собираем последовательности event_id по каждому block_id
        sequences = []
        block_ids = []
        for block_id, group in df.groupby("block_id"):
            events = group["event_id"].tolist()
            if len(events) < self.min_sequence_len:
                continue
            sequences.append(events)
            block_ids.append(block_id)

        # Если нужны только полные последовательности, возвращаем их без окон
        if not self.return_windows:
            return pd.DataFrame({"block_id": block_ids, "sequence": sequences})

        # Формируем скользящие окна фиксированного размера + target (next event)
        windows = []
        targets = []
        window_block_ids = []
        for block_id, seq in zip(block_ids, sequences):
            if len(seq) <= self.window_size:
                continue
            for i in range(0, len(seq) - self.window_size, self.stride):
                windows.append(seq[i : i + self.window_size])
                window_block_ids.append(block_id)
                if self.return_next_event:
                    targets.append(seq[i + self.window_size])

        # Собираем итоговый DataFrame
        data = {"block_id": window_block_ids, "window": windows}
        if self.return_next_event:
            data["target"] = targets
        return pd.DataFrame(data)

    def get_levels(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match["Level"] if match else None

    def get_cluster_id(self, row):
        row = str(row).strip()
        content = self.get_content(row)
        cluster = self.template_miner.match(content)
        return cluster.cluster_id if cluster else "unknown"

    def get_date(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match["Date"] if match else None

    def get_time(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match["Time"] if match else None

    def get_content(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        if match:
            return match["Content"]
        return row

    def get_block_id(self, row):
        content = self.get_content(row)
        match = self.block_id_re.search(content)
        if match:
            return match.group(1)
        if self.drop_unknown_block:
            return None
        return self.unknown_block_token

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.drain_state,
                self.drain_config,
                self.message_col,
                self.timestamp_format,
                self.block_id_regex,
                self.window_size,
                self.stride,
                self.min_sequence_len,
                self.return_windows,
                self.return_next_event,
                self.update_drain_on_fit,
                self.drop_unknown_block,
                self.unknown_block_token,
            ),
            self.__dict__,
        )
