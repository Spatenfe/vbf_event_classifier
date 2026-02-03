from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import joblib
import os

@Registry.register_method("polynomial_lr")
class PolynomialLRMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        
        # Extract params for steps
        poly_degree = self.params.get("degree", 2)
        lr_C = self.params.get("C", 1.0)
        lr_max_iter = self.params.get("max_iter", 100)
        
        self.model = Pipeline([
            ("poly", PolynomialFeatures(degree=poly_degree)),
            ("clf", LogisticRegression(C=lr_C, max_iter=lr_max_iter))
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
