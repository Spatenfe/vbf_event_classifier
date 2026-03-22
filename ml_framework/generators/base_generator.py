"""Abstract base class for all data generators."""
from abc import ABC, abstractmethod
import json
import os
import inspect
import logging


class BaseGenerator(ABC):
    def __init__(self, config):
        self.config = config
        self.name = config.get("name", "UnknownGenerator")
        self.params = config.get("params", {})

    @abstractmethod
    def fit(self, X, feature_names=None):
        """Fit the generator on a numpy array X (already filtered to target class)."""
        pass

    @abstractmethod
    def generate(self, n_samples):
        """Return a numpy array of shape (n_samples, n_features)."""
        pass

    @abstractmethod
    def save(self, output_dir):
        pass

    @abstractmethod
    def load(self, model_dir):
        pass

    @classmethod
    def get_default_config(cls):
        try:
            class_dir = os.path.dirname(os.path.abspath(inspect.getfile(cls)))
            config_path = os.path.join(class_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    return json.load(f)
        except Exception as e:
            logging.getLogger(__name__).warning("Could not load default config for %s: %s", cls.__name__, e)
        return {"name": cls.__name__, "params": {}}
