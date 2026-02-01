from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

@Registry.register_method("random_forest")
class RandomForestMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        X_test, _ = test_data
        return self.model.predict(X_test)

    def save(self, output_dir):
        joblib.dump(self.model, os.path.join(output_dir, "model.pkl"))

    def load(self, model_path):
        self.model = joblib.load(model_path)
