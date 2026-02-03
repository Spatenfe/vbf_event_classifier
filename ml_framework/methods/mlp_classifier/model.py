from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.neural_network import MLPClassifier
import joblib
import os

from sklearn.preprocessing import LabelEncoder

@Registry.register_method("mlp_classifier")
class MLPMethod(BaseAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.le = None

    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        
        # Encode labels to integers to avoid issues with early_stopping validation
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y_train)
        
        self.model = MLPClassifier(**self.params)
        self._apply_n_jobs(self.model)
        self.model.fit(X_train, y_encoded)

    def predict(self, test_data):
        X_test, _ = test_data
        preds = self.model.predict(X_test)
        # Decode back to original labels
        if self.le:
            return self.le.inverse_transform(preds)
        return preds

    def save(self, output_dir):
        # Save both model and label encoder
        data = {
            "model": self.model,
            "le": self.le
        }
        joblib.dump(data, os.path.join(output_dir, "model.pkl"))

    def load(self, model_path):
        data = joblib.load(model_path)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.le = data.get("le")
        else:
            # Fallback for old models w/o le
            self.model = data
            self.le = None
