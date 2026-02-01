import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None

    def load_data(self):
        logging.info(
            f"Loading data from train: {self.config.get('train_path')} and val: {self.config.get('val_path')}"
        )
        try:
            # Use pre-split files if available, otherwise fall back to original (though config implies split exists)
            train_df = pd.read_csv(self.config["train_path"])
            val_df = pd.read_csv(self.config["val_path"])

            # Drop columns
            if "drop_columns" in self.config:
                train_df.drop(
                    columns=self.config["drop_columns"], inplace=True, errors="ignore"
                )
                val_df.drop(
                    columns=self.config["drop_columns"], inplace=True, errors="ignore"
                )

            target_col = self.config["target_column"]

            self.X_train = train_df.drop(columns=[target_col])
            self.y_train = train_df[target_col].astype(str)

            self.X_test = val_df.drop(columns=[target_col])
            self.y_test = val_df[target_col].astype(str)

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def apply_sampling(self):
        sampling_strategy = self.config.get("sampling", {}).get("strategy", "none")

        if sampling_strategy == "none":
            return

        logging.info(f"Applying sampling strategy: {sampling_strategy}")
        sampler = None

        if sampling_strategy == "random_oversample":
            sampler = RandomOverSampler(
                random_state=self.config.get("random_state", 42)
            )
        elif sampling_strategy == "random_undersample":
            sampler = RandomUnderSampler(
                random_state=self.config.get("random_state", 42)
            )
        elif sampling_strategy == "smote":
            sampler = SMOTE(random_state=self.config.get("random_state", 42))

        if sampler:
            self.X_train, self.y_train = sampler.fit_resample(
                self.X_train, self.y_train
            )
            logging.info(f"Resampled train shape: {self.X_train.shape}")

    def preprocess(self):
        logging.info("Preprocessing data...")

        # Apply sampling first (on training data only)
        self.apply_sampling()

        # Feature Engineering / Scaling
        scaling_type = self.config.get("preprocessing", {}).get("scaling")
        if scaling_type == "standard":
            self.scaler = StandardScaler()
        elif scaling_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        if self.scaler:
            logging.info(f"Applying {scaling_type} scaling...")
            # Fit on TRAIN, transform both
            self.X_train = pd.DataFrame(
                self.scaler.fit_transform(self.X_train), columns=self.X_train.columns
            )
            self.X_test = pd.DataFrame(
                self.scaler.transform(self.X_test), columns=self.X_test.columns
            )

        # Normalization
        norm_type = self.config.get("preprocessing", {}).get("normalization")
        if norm_type:
            logging.info(f"Applying {norm_type} normalization...")
            normalizer = Normalizer(norm=norm_type)
            self.X_train = pd.DataFrame(
                normalizer.fit_transform(self.X_train), columns=self.X_train.columns
            )
            self.X_test = pd.DataFrame(
                normalizer.transform(self.X_test), columns=self.X_test.columns
            )

        logging.info(
            f"Final Data shape: Train={self.X_train.shape}, Test={self.X_test.shape}"
        )

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
