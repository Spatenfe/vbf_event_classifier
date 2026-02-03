import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
        self._has_dedicated_test = False
        
    def load_data(self):
        params = self.config.get("params", {})
        data_path = params.get("data_path") or params.get("dataset_path")
        train_path = params.get("train_path")
        val_path = params.get("val_path")
        test_path = params.get("test_path")
        target_column = params.get("target_column", "cvv")
        drop_columns = params.get("drop_columns", [])
        drop_labels = params.get("discard_classes", params.get("drop_labels", []))
        balance_train = params.get("balance_train", False)
        balance_val = params.get("balance_val", True) # Default to true as per request logic if splitting
        val_split_ratio = params.get("val_split", 0.2)

        # Helper to filter labels
        def filter_labels(df, target_col, labels_to_drop):
            if not labels_to_drop:
                return df
            return df[~df[target_col].astype(str).isin([str(l) for l in labels_to_drop])].copy()
        
        # Logic: If data_path is provided, we split. Else fall back to train_path/val_path
        if data_path:
            print(f"StandardDataloader: Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Remove full duplicates first
            original_len = len(df)
            df.drop_duplicates(inplace=True)
            if len(df) < original_len:
                print(f"StandardDataloader: Dropped {original_len - len(df)} full duplicate rows.")
            
            if drop_columns:
                df.drop(columns=drop_columns, inplace=True, errors="ignore")
            
            df = filter_labels(df, target_column, drop_labels)
            
            X = df.drop(columns=[target_column])
            y = df[target_column].astype(str)
            
            # Split 80/20
            # We use stratify to maintain distribution in the split initially
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=val_split_ratio, stratify=y, random_state=42
            )
            
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            
            # Balance Validation Set if requested (User requirement: classes are balanced in validation set)
            if balance_val:
                # We interpret "balanced" as equal counts (undersample)
                print("Balancing validation set...")
                self.X_val, self.y_val = DataBalancer.balance(self.X_val, self.y_val, strategy="undersample", random_state=42)

        else:
            # Legacy/Dual file mode
            # Load Data
            df_train = pd.read_csv(train_path)
            
            if drop_columns:
                df_train.drop(columns=drop_columns, inplace=True, errors="ignore")
                
            df_train = filter_labels(df_train, target_column, drop_labels)
                
            self.X_train = df_train.drop(columns=[target_column])
            self.y_train = df_train[target_column].astype(str)
            
            if val_path:
                df_val = pd.read_csv(val_path)
                if drop_columns:
                    df_val.drop(columns=drop_columns, inplace=True, errors="ignore")
                
                df_val = filter_labels(df_val, target_column, drop_labels)
                
                self.X_val = df_val.drop(columns=[target_column])
                self.y_val = df_val[target_column].astype(str)

        # Runtime Balancing for Train
        if balance_train:
            print(f"Applying runtime balancing ({balance_train}) to training data...")
            self.X_train, self.y_train = DataBalancer.balance(self.X_train, self.y_train, strategy=balance_train)
        
        if test_path:
            df_test = pd.read_csv(test_path)
            if drop_columns:
                df_test.drop(columns=drop_columns, inplace=True, errors="ignore")
            
            df_test = filter_labels(df_test, target_column, drop_labels)
            
            self.X_test = df_test.drop(columns=[target_column])
            self.y_test = df_test[target_column].astype(str)
            self._has_dedicated_test = True
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
            
        # Cast to float32 for memory efficiency
        self.X_train = self.X_train.astype(np.float32)
        if self.X_val is not None: self.X_val = self.X_val.astype(np.float32)
        if self.X_test is not None: self.X_test = self.X_test.astype(np.float32)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_val_data(self):
        return self.X_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_normalization_stats(self):
        return {}

    def has_test_set(self):
        return self._has_dedicated_test
