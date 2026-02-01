from abc import ABC, abstractmethod
import json
import os
import joblib

class BaseAlgorithm(ABC):
    def __init__(self, config):
        """
        Initialize the algorithm with a configuration dictionary.
        """
        self.config = config
        self.name = self.config.get("name", "UnknownAlgorithm")
        self.params = self.config.get("params", {})
        self.model = None

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
            print(f"Warning: Could not load default config for {cls.__name__}: {e}")
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
