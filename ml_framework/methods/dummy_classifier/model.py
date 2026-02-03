from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.dummy import DummyClassifier

@Registry.register_method("dummy_classifier")
class DummyAlgorithm(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        self.model = DummyClassifier(strategy="most_frequent")
        self._apply_n_jobs(self.model)
        self.model.fit(X_train, y_train)

    def predict(self, test_data):
        X_test, _ = test_data
        return self.model.predict(X_test)

    def save(self, output_dir):
        pass

    def load(self, model_path):
        pass
