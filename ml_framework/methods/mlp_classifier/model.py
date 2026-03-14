from ml_framework.core.base_method import BaseAlgorithm
from ml_framework.core.registry import Registry
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
import numpy as np

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu', dropout_rate=0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        
        act_func = nn.ReLU() if activation == 'relu' else nn.Tanh() # Default to ReLU
            
        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_func)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 1)) # Output layer for binary classification
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

@Registry.register_method("mlp_classifier")
class MLPMethod(BaseAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        self.le = None
        self.output_dir = None

    def set_output_dir(self, output_dir):
        """Set the output directory for saving plots and artifacts."""
        self.output_dir = output_dir

    def train(self, train_data, val_data=None):
        X_train, y_train = train_data
        
        # Encode labels to integers to avoid issues with early_stopping validation
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y_train)
        
        params = self.params.copy()
        
        # We will handle early stopping and best model saving manually
        params['early_stopping'] = False
        max_iter = params.pop('max_iter', 200)
        n_iter_no_change = params.pop('n_iter_no_change', 10)
        
        # Use val_data for validation, or split if not provided
        if val_data is not None:
            X_val, y_val = val_data
            y_val_encoded = self.le.transform(y_val)
        else:
            val_fraction = params.pop('validation_fraction', 0.2)
            X_train, X_val, y_encoded, y_val_encoded = train_test_split(
                X_train, y_encoded, test_size=val_fraction, 
                random_state=params.get('random_state', 42), stratify=y_encoded
            )
            
            
        # Parse params 
        hidden_layer_sizes = params.get('hidden_layer_sizes', [100])
        activation = params.get('activation', 'relu')
        alpha = params.get('alpha', 0.0001) # L2 regularization (weight decay)
        dropout_rate = params.get('dropout_rate', 0.3)
        batch_size = params.get('batch_size', 256)
        learning_rate_init = params.get('learning_rate_init', 0.001)
        
        # Build PyTorch model
        input_dim = X_train.shape[1]
        self.pytorch_model = PyTorchMLP(
            input_dim=input_dim, 
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_model.to(device)
        
        # Optimizer and criterion
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.pytorch_model.parameters(), 
            lr=learning_rate_init, 
            weight_decay=alpha
        )
        
        # DataLoaders
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_encoded, dtype=torch.float32).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val_encoded, dtype=torch.float32).unsqueeze(1).to(device)
        
        X_train_eval_t = X_train_t.to(device) # For full train score eval
        
        best_val_score = -np.inf
        # PyTorch equivalent of saving weights
        best_state_dict = None
        epochs_no_improve = 0
        best_epoch = 0
        
        loss_curve = []
        validation_loss_curve = []
        train_scores = []
        validation_scores = []
        
        classes = np.unique(y_encoded)
        
        from sklearn.metrics import log_loss
        for epoch in range(1, max_iter + 1):
            self.pytorch_model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                
            epoch_loss /= len(train_dataset)
            loss_curve.append(epoch_loss)
            
            # Evaluate using PyTorch
            self.pytorch_model.eval()
            with torch.no_grad():
                # Train metrics
                train_probs = self.pytorch_model(X_train_eval_t).cpu().numpy()
                train_preds = (train_probs >= 0.5).astype(int).flatten()
                train_score = accuracy_score(y_encoded, train_preds)
                train_scores.append(train_score)
                
                # Val metrics
                val_probs = self.pytorch_model(X_val_t).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int).flatten()
                val_score = accuracy_score(y_val_encoded, val_preds)
                validation_scores.append(val_score)
                
                # Validation loss
                val_loss = log_loss(y_val_encoded, val_probs, labels=classes)
                validation_loss_curve.append(val_loss)
            
            # Check for improvement
            if val_score > best_val_score + 1e-4:
                best_val_score = val_score
                best_state_dict = copy.deepcopy(self.pytorch_model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= n_iter_no_change:
                print(f"Early stopping at epoch {epoch}. Best validation score: {best_val_score:.4f} (at epoch {best_epoch}).")
                break
                
        # Restore best weights
        if best_state_dict is not None:
            self.pytorch_model.load_state_dict(best_state_dict)
            
        # Store attributes for plotting compat
        self.loss_curve_ = loss_curve
        self.validation_loss_curve_ = validation_loss_curve
        self.train_scores_ = train_scores
        self.validation_scores_ = validation_scores
            
        print(f"MLP (PyTorch) training finished. Restored weights from epoch {best_epoch} with val_score {best_val_score:.4f}.")
        
        # Plot loss curves if output directory is set
        if self.output_dir:
            self._plot_loss_curves()

    def _plot_loss_curves(self):
        """Plot training and validation loss & accuracy curves."""
        if not hasattr(self, 'loss_curve_'):
            return
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        epochs = range(1, len(self.loss_curve_) + 1)
        
        # Plot Loss
        ax1.plot(epochs, self.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        if hasattr(self, 'validation_loss_curve_') and self.validation_loss_curve_:
            ax1.plot(epochs, self.validation_loss_curve_, 'r-', linewidth=2, label='Validation Loss')
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot Accuracy
        if hasattr(self, 'train_scores_') and self.train_scores_:
            ax2.plot(epochs, self.train_scores_, 'b-', linewidth=2, label='Training Accuracy')
        if hasattr(self, 'validation_scores_') and self.validation_scores_:
            ax2.plot(epochs, self.validation_scores_, 'r-', linewidth=2, label='Validation Accuracy')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_metrics.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


    def predict(self, test_data):
        X_test, _ = test_data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.pytorch_model.to(device)
        self.pytorch_model.eval()
        
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            probs = self.pytorch_model(X_test_t).cpu().numpy()
            
        preds = (probs >= 0.5).astype(int).flatten()
            
        # Decode back to original labels
        if self.le:
            return self.le.inverse_transform(preds)
        return preds

    def save(self, output_dir):
        # Save both model architecture/weights and label encoder
        
        model_path = os.path.join(output_dir, "pytorch_model.pt")
        torch.save(self.pytorch_model.state_dict(), model_path)
        
        # Save exact network dimensions/parameters used during training
        network_meta = {
            "input_dim": self.pytorch_model.network[0].in_features,
            "hidden_layer_sizes": self.params.get('hidden_layer_sizes', [100]),
            "activation": self.params.get('activation', 'relu'),
            "dropout_rate": self.params.get('dropout_rate', 0.2)
        }
        
        data = {
            "le": self.le,
            "params": self.params, # original params
            "network_meta": network_meta
        }
        joblib.dump(data, os.path.join(output_dir, "model_meta.pkl"))
        
        # Save label encoder mapping as JSON for ONNX inference
        if self.le is not None:
            import json
            label_mapping = {
                "classes": self.le.classes_.tolist()
            }
            with open(os.path.join(output_dir, "label_encoder.json"), "w") as f:
                json.dump(label_mapping, f, indent=2)

    def load(self, model_path):
        # Note: model_path could be a directory or a specific file path
        if os.path.isdir(model_path):
            output_dir = model_path
        else:
            output_dir = os.path.dirname(model_path)
            
        # Load metadata
        meta_path = os.path.join(output_dir, "model_meta.pkl")
        if os.path.exists(meta_path):
            data = joblib.load(meta_path)
            self.le = data.get("le")
            self.params = data.get("params", {})
            
            # Rebuild PyTorch model architecture if meta is available
            network_meta = data.get("network_meta")
            if network_meta:
                 self.pytorch_model = PyTorchMLP(
                     input_dim=network_meta["input_dim"],
                     hidden_layer_sizes=network_meta["hidden_layer_sizes"],
                     activation=network_meta["activation"],
                     dropout_rate=network_meta["dropout_rate"]
                 )
                 
                 # Load weights into the newly rebuilt network
                 pt_path = os.path.join(output_dir, "pytorch_model.pt")
                 if os.path.exists(pt_path):
                     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                     state_dict = torch.load(pt_path, map_location=device, weights_only=True)
                     self.pytorch_model.load_state_dict(state_dict)
        else:
            self.le = None
