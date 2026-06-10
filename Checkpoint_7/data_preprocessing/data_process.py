from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import pandas as pd
import logging
import json
import time
import os
import sys
import re
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

LOG_FILE = r"D:\study_work\hse\team_project\checkpoint_6\HDFS_v1\HDFS.log"
LOG_LABEL_FILE = r"D:\study_work\hse\team_project\checkpoint_6\HDFS_v1\preprocessed\anomaly_label.csv"

def parse_and_save():
    """
    Шаблонизация логов с помощью Drain и сохранение шаблонизированных логов в drain3_structured.csv,
    а также шаблонов в drain3_templates.csv

    Для конфигурации шаблонизатора используется файл drain3.ini
    """
    config = TemplateMinerConfig()
    config.load(f"{os.getcwd()}//drain3.ini")

    template_miner = TemplateMiner(config=config)

    line_count = 0
    pattern = re.compile(r'^(?P<Date>.+?)\s+(?P<Time>.+?)\s+(?P<Pid>.+?)\s+(?P<Level>.+?)\s+(?P<Component>.+?):\s+(?P<Content>.+?)$')

    pre_results = []

    with open(LOG_FILE) as f:
        lines = f.readlines()

    start_time = time.time()

    for line in tqdm(lines):
        line = line.rstrip()
        original_log = line
        match = re.match(pattern, original_log)
        line = line.partition(": ")[2]
        result = template_miner.add_log_message(line)
        pre_results.append((result['cluster_id'],
                            original_log,
                            line_count,
                            match.group('Date').strip(),
                            match.group('Time').strip(),
                            match.group('Level').strip()))
        if result["change_type"] != "none":
            result_json = json.dumps(result)

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters")


    cluster_logs = defaultdict(list)

    for idx, *content  in pre_results:
        cluster_logs[idx].append(content)

    result_data = []

    for cluster in template_miner.drain.clusters:
        cluster_id = cluster.cluster_id
        template = cluster.get_template()
        if cluster_id in cluster_logs:
            for log, line_id, date, log_time, level in cluster_logs[cluster_id]:
                result_data.append({
                    'line_id': line_id,
                    'date': date,
                    'time': log_time,
                    'level': level,
                    'original_message': log,
                    'cluster_id': cluster_id,
                    'template': template
                })

    df_result = pd.DataFrame(result_data)
    df_result = df_result.sort_values(by='line_id',)

    result_templates = []
    for cluster in template_miner.drain.clusters:
        result_templates.append({
            "cluster_id": cluster.cluster_id,
            "occurrencies": cluster.size,
            "template": cluster.get_template()
        })

    df_result_templates = pd.DataFrame(result_templates).sort_values(by="occurrences", ascending=False)

    df_result.to_csv('drain3_structured.csv', index=False)
    df_result_templates.to_csv("drain3_templates.csv", index=False)


def sampling(log_file):
    """
    Объединение логов по block id и сохранение в drain3_log_seq.csv
    """

    df = pd.read_csv(log_file, dtype={"Date": object, "Time": object})
    
    data_dict = defaultdict(list)

    for _, row in tqdm(df.iterrows()):
        blk_id_list = re.findall(r"(blk_-?\d+)", row["original_message"])
        blk_id_set = set(blk_id_list)
        for blk_id in blk_id_set:
            data_dict[blk_id].append(row["cluster_id"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=["block_id", "event_seq"])
    data_df.to_csv("drain3_log_seq.csv", index=False)


def generate_train_test(drain_seq_file, train_ratio=0.5, val_ration=0.1):
    """
    Создание тренировочной, валидационной и тестовой выборки из последовательностей логов и сохранение в соответствующие файлы
    """
    drain_seq_df = pd.read_csv(drain_seq_file)
    labels_df = pd.read_csv(LOG_LABEL_FILE)

    combined_df = pd.merge(drain_seq_df, labels_df, left_on="block_id", right_on="BlockId", how="outer").drop("BlockId", axis=1)

    normal_seq = combined_df[combined_df["Label"] == "Normal"]["event_seq"]
    normal_seq = normal_seq.sample(frac=1, random_state=42)

    abnormal_seq = combined_df[combined_df["Label"] == "Anomaly"]["event_seq"]
    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = int(normal_len * train_ratio)
    val_normal_len = int(normal_len * val_ration) + train_len
    val_abnormal_len = int(abnormal_len * val_ration)
    print(f"Normal size: {normal_len} | Abnormal size: {abnormal_len}")
    print(f"Train size: {train_len} | Validation normal size: {val_normal_len - train_len} | Validation abnormal size: {val_abnormal_len}")
    print(f"Test normal size: {normal_len - val_normal_len} | Test abnormal size: {abnormal_len - val_abnormal_len}")

    train = normal_seq[:train_len]
    validation_normal = normal_seq[train_len: val_normal_len]
    validation_abnormal = abnormal_seq[:val_abnormal_len]
    test_normal = normal_seq[val_normal_len:]
    test_abnormal = abnormal_seq[val_abnormal_len:]

    train.to_csv("train.csv", index=False)
    validation_normal.to_csv("validation_normal.csv", index=False)
    validation_abnormal.to_csv("validation_abnormal.csv", index=False)
    test_normal.to_csv("test_normal.csv", index=False)
    test_abnormal.to_csv("test_abnormal.csv", index=False)


def create_vocab(train_file):
    """
    Создание словаря шаблонов со специальными токенами на основе тренировочного датасета
    """
    train_df = pd.read_csv(train_file)
    event_set = set()
    for _, row in tqdm(train_df.iterrows()):
        event_set.update(set(eval(row.event_seq)))
    vocab = {"PAD": 0, "MASK": 1, "UNK": 2, "CLS": 3, "EOS": 4}
    for event_id in event_set:
        if event_id not in vocab:
            vocab[event_id] = len(vocab)

    with open("vocab.json", "w") as f:
        json.dump(vocab, f)
    return vocab


# parse_and_save()
# sampling("drain3_structured.csv")
# generate_train_test("drain3_log_seq.csv", train_ratio=0.8, val_ration=0.1)
# vocabulary = create_vocab("train.csv")
