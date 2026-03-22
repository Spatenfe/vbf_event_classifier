"""Shared base class and utilities for PyTorch-based classification methods."""
from ml_framework.core.base_method import BaseAlgorithm
import torch
import torch.nn as nn
import torch.optim as optim


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: per-channel learned importance weighting."""
    def __init__(self, channels, ratio=8):
        super().__init__()
        reduced = max(1, channels // ratio)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with severe class imbalance.
    Down-weights easy examples, focuses learning on hard boundary cases.
    FL(p) = -alpha * (1 - p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import copy
import numpy as np


class PyTorchBaseMethod(BaseAlgorithm):
    """Base class for PyTorch classifiers. Subclasses implement _build_model()."""

    def __init__(self, config):
        super().__init__(config)
        self.le = None
        self.output_dir = None
        self.pytorch_model = None

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def _build_model(self, input_dim, params):
        """Return an nn.Module. Must be implemented by subclasses."""
        raise NotImplementedError

    def _get_network_meta(self):
        """Return a dict with enough info to reconstruct the model in load()."""
        raise NotImplementedError

    def _build_model_from_meta(self, meta):
        """Reconstruct and return an nn.Module from saved metadata."""
        raise NotImplementedError

    def train(self, train_data, val_data=None):
        X_train, y_train = train_data

        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y_train)

        params = self.params.copy()
        params.pop('early_stopping', None)
        max_iter = params.pop('max_iter', 200)
        n_iter_no_change = params.pop('n_iter_no_change', 10)

        if val_data is not None:
            X_val, y_val = val_data
            y_val_encoded = self.le.transform(y_val)
        else:
            val_fraction = params.pop('validation_fraction', 0.2)
            X_train, X_val, y_encoded, y_val_encoded = train_test_split(
                X_train, y_encoded, test_size=val_fraction,
                random_state=params.get('random_state', 42), stratify=y_encoded
            )

        alpha = params.get('alpha', 0.0001)
        batch_size = params.get('batch_size', 256)
        learning_rate_init = params.get('learning_rate_init', 0.001)
        scheduler_name = params.get('lr_scheduler', None)
        label_smoothing = params.get('label_smoothing', 0.0)
        loss_fn_name = params.get('loss_fn', 'bce')

        input_dim = X_train.shape[1]
        self.pytorch_model = self._build_model(input_dim, params)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_model.to(device)

        # Loss function selection
        if loss_fn_name == 'weighted_bce':
            n_pos = max(int(np.sum(y_encoded == 1)), 1)
            n_neg = max(int(np.sum(y_encoded == 0)), 1)
            pos_weight_val = float(n_neg / n_pos)
            criterion = nn.BCELoss(reduction='none')
            print(f"Using weighted BCE: pos_weight={pos_weight_val:.2f} (n_neg={n_neg}, n_pos={n_pos})")
        elif loss_fn_name == 'focal':
            focal_gamma = params.get('focal_gamma', 2.0)
            focal_alpha = params.get('focal_alpha', 0.25)
            criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
            pos_weight_val = None
            print(f"Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
        else:
            criterion = nn.BCELoss()
            pos_weight_val = None

        optimizer = optim.Adam(
            self.pytorch_model.parameters(),
            lr=learning_rate_init,
            weight_decay=alpha
        )

        # Optional LR scheduler
        if scheduler_name == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=max(1, n_iter_no_change // 2), factor=0.5
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter)
        else:
            scheduler = None

        X_train_t = torch.tensor(np.array(X_train), dtype=torch.float32)
        y_train_t = torch.tensor(np.array(y_encoded), dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        X_val_t = torch.tensor(np.array(X_val), dtype=torch.float32).to(device)
        y_val_t = torch.tensor(np.array(y_val_encoded), dtype=torch.float32).unsqueeze(1).to(device)

        classes = np.unique(y_encoded)
        best_val_score = -np.inf
        best_state_dict = None
        epochs_no_improve = 0
        best_epoch = 0

        loss_curve = []
        validation_loss_curve = []
        train_scores = []
        validation_scores = []

        for epoch in range(1, max_iter + 1):
            self.pytorch_model.train()
            epoch_loss = 0.0
            all_train_preds = []
            all_train_labels = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)
                batch_y_smooth = batch_y * (1 - label_smoothing) + 0.5 * label_smoothing
                if pos_weight_val is not None:
                    weights = torch.where(batch_y == 1,
                                          torch.full_like(batch_y, pos_weight_val),
                                          torch.ones_like(batch_y))
                    loss = (criterion(outputs, batch_y_smooth) * weights).mean()
                else:
                    loss = criterion(outputs, batch_y_smooth)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)

                # Accumulate train predictions from this batch (no extra forward pass)
                preds = (outputs.detach() >= 0.5).int().flatten().cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(batch_y.flatten().cpu().numpy().astype(int))

            epoch_loss /= len(train_dataset)
            loss_curve.append(epoch_loss)

            train_score = accuracy_score(all_train_labels, all_train_preds)
            train_scores.append(train_score)

            self.pytorch_model.eval()
            with torch.no_grad():
                val_probs = self.pytorch_model(X_val_t).cpu().numpy()
                val_preds = (val_probs >= 0.5).astype(int).flatten()
                val_score = accuracy_score(y_val_encoded, val_preds)
                validation_scores.append(val_score)
                val_loss = log_loss(y_val_encoded, val_probs, labels=classes)
                validation_loss_curve.append(val_loss)

            # Step scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_score)
                else:
                    scheduler.step()

            if val_score > best_val_score + 1e-4:
                best_val_score = val_score
                best_state_dict = copy.deepcopy(self.pytorch_model.state_dict())
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= n_iter_no_change:
                print(f"Early stopping at epoch {epoch}. Best val score: {best_val_score:.4f} (epoch {best_epoch}).")
                break

        if best_state_dict is not None:
            self.pytorch_model.load_state_dict(best_state_dict)

        # Threshold optimization: sweep to maximize val F1-macro
        from sklearn.metrics import f1_score as _f1_score
        self.pytorch_model.eval()
        with torch.no_grad():
            val_probs_final = self.pytorch_model(X_val_t).cpu().numpy().flatten()
        best_thresh, best_f1 = 0.5, -np.inf
        for t in np.arange(0.05, 0.95, 0.01):
            f1 = _f1_score(y_val_encoded, (val_probs_final >= t).astype(int),
                           average='macro', zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, float(t)
        self.threshold_ = best_thresh
        print(f"Optimal threshold: {self.threshold_:.2f} (val F1-macro: {best_f1:.4f})")

        self.loss_curve_ = loss_curve
        self.validation_loss_curve_ = validation_loss_curve
        self.train_scores_ = train_scores
        self.validation_scores_ = validation_scores

        print(f"{self.name} training finished. Restored weights from epoch {best_epoch} with val_score {best_val_score:.4f}.")

        if self.output_dir:
            self._plot_loss_curves()

    def _plot_loss_curves(self):
        if not hasattr(self, 'loss_curve_'):
            return

        os.makedirs(self.output_dir, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(self.loss_curve_) + 1)

        ax1.plot(epochs, self.loss_curve_, 'b-', linewidth=2, label='Training Loss')
        if self.validation_loss_curve_:
            ax1.plot(epochs, self.validation_loss_curve_, 'r-', linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        if self.train_scores_:
            ax2.plot(epochs, self.train_scores_, 'b-', linewidth=2, label='Training Accuracy')
        if self.validation_scores_:
            ax2.plot(epochs, self.validation_scores_, 'r-', linewidth=2, label='Validation Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def predict(self, test_data):
        X_test, _ = test_data
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_model.to(device)
        self.pytorch_model.eval()

        X_test_t = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = self.pytorch_model(X_test_t).cpu().numpy()

        threshold = getattr(self, 'threshold_', 0.5)
        preds = (probs >= threshold).astype(int).flatten()
        if self.le is not None:
            return self.le.inverse_transform(preds)
        return preds

    def save(self, output_dir):
        torch.save(self.pytorch_model.state_dict(), os.path.join(output_dir, "pytorch_model.pt"))

        data = {
            "le": self.le,
            "params": self.params,
            "network_meta": self._get_network_meta(),
            "threshold": getattr(self, 'threshold_', 0.5),
        }
        joblib.dump(data, os.path.join(output_dir, "model_meta.pkl"))

        if self.le is not None:
            import json
            with open(os.path.join(output_dir, "label_encoder.json"), "w") as f:
                json.dump({"classes": self.le.classes_.tolist()}, f, indent=2)

    def load(self, model_path):
        output_dir = model_path if os.path.isdir(model_path) else os.path.dirname(model_path)
        meta_path = os.path.join(output_dir, "model_meta.pkl")

        if os.path.exists(meta_path):
            data = joblib.load(meta_path)
            self.le = data.get("le")
            self.params = data.get("params", {})
            self.threshold_ = data.get("threshold", 0.5)
            network_meta = data.get("network_meta")
            if network_meta:
                self.pytorch_model = self._build_model_from_meta(network_meta)
                pt_path = os.path.join(output_dir, "pytorch_model.pt")
                if os.path.exists(pt_path):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    state_dict = torch.load(pt_path, map_location=device, weights_only=True)
                    self.pytorch_model.load_state_dict(state_dict)
        else:
            self.le = None
