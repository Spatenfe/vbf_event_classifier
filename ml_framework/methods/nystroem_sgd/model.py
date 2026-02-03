from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

@Registry.register_method("nystroem_sgd")
class NystroemSGDMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        
        n_components = self.params.get("n_components", 100)
        gamma = self.params.get("gamma", None) # Default to 1/n_features
        alpha = self.params.get("alpha", 0.0001)
        max_iter = self.params.get("max_iter", 1000)
        
        self.model = Pipeline([
            ("nystroem", Nystroem(n_components=n_components, gamma=gamma, random_state=42)),
            ("clf", SGDClassifier(alpha=alpha, max_iter=max_iter, random_state=42))
        ])
        self._apply_n_jobs(self.model)
        
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        X_test, _ = test_data
        return self.model.predict(X_test)

    def save(self, output_dir):
        joblib.dump(self.model, os.path.join(output_dir, "model.pkl"))

    def load(self, model_path):
        self.model = joblib.load(model_path)
