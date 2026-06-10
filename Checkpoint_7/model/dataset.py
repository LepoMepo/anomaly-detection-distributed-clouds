from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import random
import ast
import json
import pandas as pd


class LogBERTDataset(Dataset):
    def __init__(self, vocab, sequences, seq_len=128, mask_ratio=0.15, predict_mode=False):
        super().__init__()
        self.vocab = vocab
        self.sequences = sequences
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.predict_mode = predict_mode

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        sequence = self.sequences[index]

        sequence = self.convert_event_ids_to_tokens(sequence)

        if self.predict_mode:
            masked_seq = sequence
            labels = torch.tensor(sequence, dtype=torch.long)
        else:
            masked_seq, labels = self.mask_sequence(sequence)
            labels = torch.tensor(labels, dtype=torch.long)

            for i, token in enumerate(sequence):
                if token == self.vocab["PAD"]:
                    labels[i] = -100

        return {
            "input_ids": torch.tensor(masked_seq, dtype=torch.long),
            "labels": labels
        }


    def mask_sequence(self, sequence):
        """
        Маскирование данных для обучения и предсказания. Маскирование только значимых токенов, все служебные токены пропускаются
        """
        mask_id = self.vocab["MASK"]

        masked_seq = sequence.copy()

        labels = [-100] * len(sequence)

        for i, token in enumerate(sequence):
            if token in (self.vocab["PAD"], self.vocab["CLS"], self.vocab["EOS"]):
                continue

            if random.random() < self.mask_ratio:
                labels[i] = token
                masked_seq[i] = mask_id

        return masked_seq, labels
    
    def convert_event_ids_to_tokens(self, sequence):
        """
        Конвертирование id шаблонов в токены
        """
        converted = []
        for event_id in sequence:
            key = str(event_id)
            if key in self.vocab:
                converted.append(self.vocab[key])
            else:
                converted.append(self.vocab["UNK"])

        return converted
    
    def collate_fn(self, batch):
        """
        Метод для формирования батча, реализует динамический паддинг
        """
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]

        max_len = max(len(seq) for seq in input_ids)

        padded_input_ids = torch.full((len(batch), max_len), self.vocab["PAD"], dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
            seq_len = len(ids)

            padded_input_ids[i, :seq_len] = ids
            padded_labels[i, :seq_len] = lbls

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels
        }


def load_vocab(vocab_path):
    """
    Загрузка словаря
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    return vocab


def load_log_data(data_path):
    """
    Загрузка последовательностей логов
    """
    df = pd.read_csv(data_path)
    sequences = []

    for seq in df["event_seq"]:
        seq = ast.literal_eval(seq)
        sequences.append(seq)

    return sequences


# vocab = load_vocab("vocab.json")
# logs = load_log_data("train.csv")