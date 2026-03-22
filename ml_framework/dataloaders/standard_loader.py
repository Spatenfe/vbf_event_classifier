import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, MaxAbsScaler, RobustScaler, PowerTransformer, QuantileTransformer
from ml_framework.core.base_dataloader import BaseDataloader
from ml_framework.core.registry import Registry
from ml_framework.core.data_utils import DataBalancer


logger = logging.getLogger(__name__)

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
        self.feature_names = None  # Store feature names
        self.X_train_real = None   # Real-only train data (set when augment_path is used)
        self.y_train_real = None
        
    def load_data(self):
        params = self.config.get("params", {})
        data_path = params.get("data_path") or params.get("dataset_path")
        train_path = params.get("train_path")
        val_path = params.get("val_path")
        test_path = params.get("test_path")
        target_column = params.get("target_column", "cvv")
        drop_columns = params.get("drop_columns", [])
        drop_labels = params.get("discard_classes", params.get("drop_labels", []))
        merge_classes = params.get("merge_classes", [])
        balance_train = params.get("balance_train", False)
        balance_val = params.get("balance_val", True) # Default to true as per request logic if splitting
        val_split_ratio = params.get("val_split", 0.2)

        # Helper to merge labels
        def apply_class_merging(df, target_col, merge_config):
            return DataBalancer.merge_and_balance_classes(df, target_col, merge_config)

        # Helper to filter labels
        def filter_labels(df, target_col, labels_to_drop):
            if not labels_to_drop:
                return df
            return df[~df[target_col].astype(str).isin([str(l) for l in labels_to_drop])].copy()
        
        # Logic: If data_path is provided, we split. Else fall back to train_path/val_path
        if data_path:
            logger.info("StandardDataloader: Loading data from %s", data_path)
            df = pd.read_csv(data_path)
            
            # Remove full duplicates first
            original_len = len(df)
            df.drop_duplicates(inplace=True)
            if len(df) < original_len:
                logger.info("StandardDataloader: Dropped %s full duplicate rows.", original_len - len(df))
            
            if drop_columns:
                df.drop(columns=drop_columns, inplace=True, errors="ignore")
            
            df = apply_class_merging(df, target_column, merge_classes)
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
                logger.info("Balancing validation set...")
                self.X_val, self.y_val = DataBalancer.balance(self.X_val, self.y_val, strategy="undersample", random_state=42)

        else:
            # Legacy/Dual file mode
            # Load Data
            df_train = pd.read_csv(train_path)
            
            if drop_columns:
                df_train.drop(columns=drop_columns, inplace=True, errors="ignore")
                
            df_train = apply_class_merging(df_train, target_column, merge_classes)
            df_train = filter_labels(df_train, target_column, drop_labels)
                
            self.X_train = df_train.drop(columns=[target_column])
            self.y_train = df_train[target_column].astype(str)
            
            if val_path:
                df_val = pd.read_csv(val_path)
                if drop_columns:
                    df_val.drop(columns=drop_columns, inplace=True, errors="ignore")
                df_val = apply_class_merging(df_val, target_column, merge_classes)
                df_val = filter_labels(df_val, target_column, drop_labels)
                
                self.X_val = df_val.drop(columns=[target_column])
                self.y_val = df_val[target_column].astype(str)

        # Runtime Balancing for Train
        if balance_train:
            oversample_factor = params.get("oversample_factor")
            logger.info("Applying runtime balancing (%s) to training data...", balance_train)
            if oversample_factor:
                 logger.info("Oversampling factor: %s", oversample_factor)
            self.X_train, self.y_train = DataBalancer.balance(
                self.X_train, 
                self.y_train, 
                strategy=balance_train, 
                oversample_factor=oversample_factor
            )
        
        # Synthetic augmentation: append generated samples to train set only
        augment_path = params.get("augment_path")
        _n_real_train = 0  # 0 means no augmentation was applied
        if augment_path:
            logger.info("Loading synthetic augmentation data from %s", augment_path)
            df_aug = pd.read_csv(augment_path)
            X_aug = df_aug.drop(columns=[target_column])
            y_aug = df_aug[target_column].astype(str)

            # Align columns to the real train set (generated data is a feature subset)
            train_cols = list(self.X_train.columns)
            X_aug = X_aug.reindex(columns=train_cols)

            _n_real_train = len(self.X_train)  # record split point before merging
            self.X_train = pd.concat([self.X_train, X_aug], ignore_index=True)
            self.y_train = pd.concat([self.y_train, y_aug], ignore_index=True)
            logger.info(
                "Appended %d synthetic samples. Train set: %d real + %d synthetic = %d total.",
                len(df_aug), _n_real_train, len(df_aug), len(self.X_train),
            )

        if test_path:
            df_test = pd.read_csv(test_path)
            if drop_columns:
                df_test.drop(columns=drop_columns, inplace=True, errors="ignore")
            df_test = apply_class_merging(df_test, target_column, merge_classes)
            df_test = filter_labels(df_test, target_column, drop_labels)
            
            self.X_test = df_test.drop(columns=[target_column])
            self.y_test = df_test[target_column].astype(str)
            self._has_dedicated_test = True
        
        # Preprocessing (snapshot for fine-tuning is taken inside, before noise)
        self._preprocess(params)

    def _preprocess(self, params):
        # Store feature names before preprocessing (which converts to numpy)
        if hasattr(self.X_train, 'columns'):
            self.feature_names = list(self.X_train.columns)
        
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
            
        if normalization and normalization != "none":
            normalizer = Normalizer(norm=normalization)
            self.X_train = normalizer.fit_transform(self.X_train)
            if self.X_val is not None: self.X_val = normalizer.transform(self.X_val)

            if self.X_test is not None: self.X_test = normalizer.transform(self.X_test)
            
        # Cast to float32 for memory efficiency
        self.X_train = self.X_train.astype(np.float32)
        if self.X_val is not None: self.X_val = self.X_val.astype(np.float32)
        if self.X_test is not None: self.X_test = self.X_test.astype(np.float32)

        # Snapshot scaled+normalised train data (real + VAE, no Gaussian noise) for fine-tuning phase
        self.X_train_real = self.X_train.copy()
        self.y_train_real = np.asarray(self.y_train)

        # Relative Gaussian noise augmentation (train only)
        augmentation = params.get("augmentation", {})
        noise_std = augmentation.get("noise_std") if augmentation else None
        if noise_std:
            rng = np.random.default_rng(42)
            minority_only = augmentation.get("minority_only", False)
            if minority_only:
                y_arr = np.asarray(self.y_train)
                counts = {cls: (y_arr == cls).sum() for cls in np.unique(y_arr)}
                minority_cls = min(counts, key=counts.get)
                mask = y_arr == minority_cls
                noise = rng.normal(0.0, noise_std, self.X_train[mask].shape).astype(np.float32)
                self.X_train[mask] = self.X_train[mask] * (1.0 + noise)
                logger.info("Applied relative Gaussian noise augmentation (std=%.4f) to minority class '%s' only.", noise_std, minority_cls)
            else:
                noise = rng.normal(0.0, noise_std, self.X_train.shape).astype(np.float32)
                self.X_train = self.X_train * (1.0 + noise)
                logger.info("Applied relative Gaussian noise augmentation (std=%.4f) to training data.", noise_std)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_real_train_data(self):
        """Returns scaled train data without Gaussian noise (includes real + VAE rows).
        Used for the fine-tuning phase. Returns None if preprocessing hasn't run yet."""
        if self.X_train_real is None:
            return None
        return self.X_train_real, self.y_train_real

    def get_val_data(self):
        if self.X_val is None:
            return None
        return self.X_val, self.y_val

    def get_test_data(self):
        if self.X_test is None:
            return None
        return self.X_test, self.y_test

    def get_normalization_stats(self):
        return {}

    def has_test_set(self):
        return self._has_dedicated_test
