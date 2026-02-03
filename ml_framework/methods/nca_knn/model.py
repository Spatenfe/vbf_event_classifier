from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

@Registry.register_method("nca_knn")
class NCAMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data

        requested_components = self.params.get("n_components", 2)
        requested_neighbors = self.params.get("n_neighbors", 5)

        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        # NeighborhoodComponentsAnalysis requires n_components <= n_features.
        n_components = max(1, min(int(requested_components), int(n_features)))
        # KNN requires n_neighbors <= n_samples.
        n_neighbors = max(1, min(int(requested_neighbors), int(n_samples)))
        
        self.model = Pipeline([
            ("nca", NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors))
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
