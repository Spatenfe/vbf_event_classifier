from abc import ABC, abstractmethod
import json
import os
import joblib
import logging


def _extract_n_jobs(params: dict):
    if not isinstance(params, dict):
        return None, {}
    n_jobs = params.get("n_jobs")
    filtered = {k: v for k, v in params.items() if k != "n_jobs"}
    return n_jobs, filtered

class BaseAlgorithm(ABC):
    def __init__(self, config):
        """
        Initialize the algorithm with a configuration dictionary.
        """
        self.config = config
        self.name = self.config.get("name", "UnknownAlgorithm")
        n_jobs, params = _extract_n_jobs(self.config.get("params", {}))
        # Optional: number of threads for sklearn estimators that support it.
        # Safe for all methods: ignored if underlying estimator has no n_jobs.
        self.n_jobs = n_jobs
        self.params = params
        self.model = None

    def _apply_n_jobs(self, estimator):
        """Apply self.n_jobs to estimator/pipeline if it supports n_jobs.

        This supports nested estimators (e.g., Pipeline) by setting all params
        that are either 'n_jobs' or end with '__n_jobs'.
        """
        if self.n_jobs is None or estimator is None:
            return
        if not (hasattr(estimator, "get_params") and hasattr(estimator, "set_params")):
            return

        try:
            params = estimator.get_params(deep=True)
        except Exception:
            try:
                params = estimator.get_params()
            except Exception:
                return

        n_jobs_keys = [k for k in params.keys() if k == "n_jobs" or k.endswith("__n_jobs")]
        if not n_jobs_keys:
            return

        estimator.set_params(**{k: self.n_jobs for k in n_jobs_keys})

    @classmethod
    def get_default_config(cls):
        """
        Return a default configuration dictionary loaded from config.json in the class directory.
        """
        # Get the directory where the class is defined
        import sys
        import inspect
        
        try:
            # Inspection is more reliable for finding the file where the class is defined
            class_file = inspect.getfile(cls)
            class_dir = os.path.dirname(os.path.abspath(class_file))
            config_path = os.path.join(class_dir, "config.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    return json.load(f)
            else:
                return {
                    "name": cls.__name__,
                    "params": {}
                }
        except Exception as e:
            # Fallback
            logging.getLogger(__name__).warning(
                "Could not load default config for %s: %s",
                cls.__name__,
                e,
            )
            return {
                "name": cls.__name__,
                "params": {}
            }

    @abstractmethod
    def train(self, train_data, val_data=None):
        """
        Train the model.
        train_data: Output from Dataloader.get_train_data()
        """
        pass

    @abstractmethod
    def predict(self, test_data):
        """
        Make predictions.
        test_data: Output from Dataloader.get_test_data()
        Returns: predictions
        """
        pass

    @abstractmethod
    def save(self, output_dir):
        """Save the trained model to disk."""
        pass

    @abstractmethod
    def load(self, model_path):
        """Load a trained model from disk."""
        pass
