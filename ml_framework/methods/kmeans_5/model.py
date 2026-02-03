from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.cluster import KMeans
import joblib
import os
import numpy as np

@Registry.register_method("kmeans_5")
class KMeansMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        n_clusters = self.params.get("n_clusters", 5)
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self._apply_n_jobs(self.model)
        self.model.fit(X_train)
        
        # Map clusters to labels
        # Predict clusters for training data
        train_clusters = self.model.predict(X_train)
        
        self.cluster_map = {}
        unique_blobs = np.unique(train_clusters)
        
        for blob in unique_blobs:
            # Get indices of points in this cluster
            indices = np.where(train_clusters == blob)[0]
            # Get true labels for these points. y_train might be a Series or array.
            if hasattr(y_train, 'iloc'):
                true_labels = y_train.iloc[indices]
            else:
                true_labels = y_train[indices]
                
            if len(true_labels) > 0:
                # Find most frequent label
                if hasattr(true_labels, 'mode'):
                    # pandas
                    modes = true_labels.mode()
                    if not modes.empty:
                        most_common = modes.iloc[0]
                    else:
                         most_common = true_labels.iloc[0] # Fallback
                else:
                    # numpy
                    values, counts = np.unique(true_labels, return_counts=True)
                    most_common = values[np.argmax(counts)]
                
                self.cluster_map[blob] = str(most_common)
            else:
                self.cluster_map[blob] = "unknown"

    def predict(self, test_data):
        X_test, _ = test_data
        clusters = self.model.predict(X_test)
        # Map clusters to labels
        predictions = [self.cluster_map.get(c, "unknown") for c in clusters]
        return np.array(predictions)

    def save(self, output_dir):
        joblib.dump(self.model, os.path.join(output_dir, "model.pkl"))

    def load(self, model_path):
        self.model = joblib.load(model_path)
