from sklearn.base import BaseEstimator, TransformerMixin
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence
import re
import pandas as pd
import joblib


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drain_state, drain_config, time_window='30s'):
        self.time_window = time_window
        self.drain_state = drain_state
        self.drain_config = drain_config
        persistance = FilePersistence(self.drain_state)
        config = TemplateMinerConfig()
        config.load(self.drain_config)
        self.template_miner = TemplateMiner(persistence_handler=persistance, config=config)
        self.pattern = re.compile(
            r'^(?P<Date>.+?)\s+(?P<Time>.+?)\s+(?P<Pid>.+?)\s+(?P<Level>.+?)\s+(?P<Component>.+?):\s+(?P<Content>.+?)$',
            re.IGNORECASE)
        self.template_list_ = None
        self.level_list_ = None

    def fit(self, X, y=None):
        self.template_list_ = [
            cluster.cluster_id
            for cluster in self.template_miner.drain.clusters
        ]
        self.template_list_.append('unknown')

        self.level_list_ = list(X['original_message'].apply(self.get_levels).unique())

        return self

    def transform(self, X):
        df_transformed = pd.DataFrame()

        df_transformed['date'] = X['original_message'].apply(self.get_date)
        df_transformed['time'] = X['original_message'].apply(self.get_time)
        df_transformed['level'] = X['original_message'].apply(self.get_levels)
        df_transformed['cluster_id'] = X['original_message'].apply(self.get_cluster_id)

        df_transformed['timestamp'] = df_transformed['date'].astype(str) + ' ' + df_transformed['time'].astype(str)
        df_transformed['timestamp'] = pd.to_datetime(df_transformed['timestamp'], format='%y%m%d %H%M%S')
        df_transformed['interval'] = df_transformed['timestamp'].dt.floor(self.time_window)

        df_transformed = df_transformed.drop(['date', 'time'], axis=1)

        df_features = df_transformed.pivot_table(
            index='interval',
            columns='cluster_id',
            aggfunc='size',
            fill_value=0
        ).reindex(columns=self.template_list_, fill_value=0)

        df_ohe_features = df_transformed.pivot_table(
            index='interval',
            columns='level',
            aggfunc='size',
            fill_value=0
        ).reindex(columns=self.level_list_, fill_value=0)

        df_result = pd.concat([df_features,
                               df_ohe_features],
                              axis=1)

        df_result.columns = df_result.columns.astype(str)
        return df_result

    def get_levels(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match['Level']

    def get_cluster_id(self, row):
        row = str(row).strip()
        row = row.partition(': ')[2]
        cluster = self.template_miner.match(row)
        return cluster.cluster_id if cluster else 'unknown'

    def get_date(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match['Date']

    def get_time(self, row):
        row = str(row).strip()
        match = re.match(self.pattern, row)
        return match['Time']

    def __reduce__(self):
        return self.__class__, (self.drain_state, self.drain_config, self.time_window), self.__dict__

