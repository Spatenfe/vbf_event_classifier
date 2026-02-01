from ml_framework.core.base_dataloader import BaseDataloader
from ml_framework.core.registry import Registry
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

@Registry.register_dataloader("simple_loader")
class SimpleDataloader(BaseDataloader):
    def __init__(self, config):
        super().__init__(config)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_normalization_stats(self):
        return {}
