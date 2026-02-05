from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

@Registry.register_method("xgb_classifier")
class XGBMethod(BaseAlgorithm):
    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        
        # XGBoost requires labels to be integers from 0 to n_classes-1
        self.encoder = LabelEncoder()
        y_train_encoded = self.encoder.fit_transform(y_train)
        
        self.model = XGBClassifier(**self.params)
        self.model.fit(X_train, y_train_encoded)

    def predict(self, test_data):
        X_test, _ = test_data
        preds_encoded = self.model.predict(X_test)
        return self.encoder.inverse_transform(preds_encoded)

    def save(self, output_dir):
        joblib.dump(self.model, os.path.join(output_dir, "model.pkl"))
        joblib.dump(self.encoder, os.path.join(output_dir, "encoder.pkl"))

    def load(self, model_path):
        self.model = joblib.load(model_path)
        # Assuming encoder is in the same directory as model_path
        encoder_path = os.path.join(os.path.dirname(model_path), "encoder.pkl")
        if os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
        else:
            # Fallback or warning if loading an old model without encoder (shouldn't happen for new code)
            pass
