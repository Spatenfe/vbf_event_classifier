from abc import ABC, abstractmethod

class BaseDataloader(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load_data(self):
        """Load and prepare data."""
        pass

    @abstractmethod
    def get_train_data(self):
        """Return training data."""
        pass

    @abstractmethod
    def get_test_data(self):
        """Return testing data."""
        pass

    @abstractmethod
    def get_normalization_stats(self):
        """Return normalization statistics if applicable."""
        pass
