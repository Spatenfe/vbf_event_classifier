import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from ml_framework.core.base_dataloader import BaseDataloader
from ml_framework.core.registry import Registry
from ml_framework.core.data_utils import DataBalancer

@Registry.register_dataloader("standard_loader")
class StandardDataloader(BaseDataloader):
    def __init__(self, config):
        super().__init__(config)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
    def load_data(self):
        params = self.config.get("params", {})
        train_path = params.get("train_path")
        val_path = params.get("val_path")
        test_path = params.get("test_path") # Optional, might use val as test
        target_column = params.get("target_column", "cvv")
        drop_columns = params.get("drop_columns", [])
        drop_labels = params.get("drop_labels", [])
        balance_train = params.get("balance_train", False)
        
        # Helper to filter labels
        def filter_labels(df, target_col, labels_to_drop):
            if not labels_to_drop:
                return df
            return df[~df[target_col].astype(str).isin([str(l) for l in labels_to_drop])].copy()
        
        # Load Data
        df_train = pd.read_csv(train_path)
        
        if drop_columns:
            df_train.drop(columns=drop_columns, inplace=True, errors="ignore")
            
        df_train = filter_labels(df_train, target_column, drop_labels)
            
        self.X_train = df_train.drop(columns=[target_column])
        self.y_train = df_train[target_column].astype(str)
        
        # Runtime Balancing
        if balance_train:
            print(f"Applying runtime balancing ({balance_train}) to training data...")
            self.X_train, self.y_train = DataBalancer.balance(self.X_train, self.y_train, strategy=balance_train)
        
        if val_path:
            df_val = pd.read_csv(val_path)
            if drop_columns:
                df_val.drop(columns=drop_columns, inplace=True, errors="ignore")
            
            df_val = filter_labels(df_val, target_column, drop_labels)
            
            self.X_val = df_val.drop(columns=[target_column])
            self.y_val = df_val[target_column].astype(str)
        
        if test_path:
            df_test = pd.read_csv(test_path)
            if drop_columns:
                df_test.drop(columns=drop_columns, inplace=True, errors="ignore")
            
            df_test = filter_labels(df_test, target_column, drop_labels)
            
            self.X_test = df_test.drop(columns=[target_column])
            self.y_test = df_test[target_column].astype(str)
        elif self.X_val is not None:
             # Use val as test if test not provided
            self.X_test = self.X_val
            self.y_test = self.y_val
            
        # Preprocessing
        self._preprocess(params)

    def _preprocess(self, params):
        preprocessing = params.get("preprocessing", {})
        scaling = preprocessing.get("scaling")
        normalization = preprocessing.get("normalization")
        
        if scaling == "standard":
            scaler = StandardScaler()
        elif scaling == "minmax":
            scaler = MinMaxScaler()
        elif scaling == "maxabs":
            scaler = MaxAbsScaler()
        elif scaling == "robust":
            scaler = RobustScaler()
        elif scaling == "yeo-johnson":
            scaler = PowerTransformer(method="yeo-johnson")
        elif scaling == "quantile_normal":
            scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        elif scaling == "quantile_uniform":
            scaler = QuantileTransformer(output_distribution="uniform", random_state=42)
        else:
            scaler = None
            
        if scaler:
            self.X_train = scaler.fit_transform(self.X_train)
            if self.X_val is not None: self.X_val = scaler.transform(self.X_val)
            if self.X_test is not None: self.X_test = scaler.transform(self.X_test)
            
        if normalization:
            normalizer = Normalizer(norm=normalization)
            self.X_train = normalizer.fit_transform(self.X_train)
            if self.X_val is not None: self.X_val = normalizer.transform(self.X_val)
            if self.X_test is not None: self.X_test = normalizer.transform(self.X_test)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_normalization_stats(self):
        return {}
