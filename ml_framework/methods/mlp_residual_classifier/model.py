from ml_framework.core.registry import Registry
from ml_framework.methods.pytorch_base import PyTorchBaseMethod, SEBlock
from ml_framework.methods.mlp_classifier.model import _ACT_MAP
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Two-layer residual block: Linear -> [BN] -> Act -> Dropout -> Linear -> [BN] + shortcut -> Act.
    A projection shortcut is used when in_dim != out_dim.
    """
    def __init__(self, in_dim, out_dim, activation_cls, dropout_rate, batch_norm, se_ratio):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.act = activation_cls()
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.se = SEBlock(out_dim, ratio=se_ratio) if se_ratio > 0 else nn.Identity()
        self.shortcut = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        out = self.drop(self.act(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out = self.se(out)
        return self.act(out + self.shortcut(x))


class PyTorchResidualMLP(nn.Module):
    """
    MLP with residual blocks. `hidden_layer_sizes` defines the output dimension of each block.
    The first entry also serves as the stem output dimension.

    Example with hidden_layer_sizes=[256, 128, 64]:
        Stem:  Linear(in → 256) + [BN] + Act + Dropout
        Block: ResidualBlock(256 → 128)
        Block: ResidualBlock(128 → 64)
        Head:  Linear(64 → 1) + Sigmoid
    """
    def __init__(self, input_dim, hidden_layer_sizes=(256, 128, 64), activation='relu',
                 dropout_rate=0.3, batch_norm=True, se_ratio=0):
        super().__init__()
        act_cls = _ACT_MAP.get(activation, nn.ReLU)

        stem_dim = hidden_layer_sizes[0]
        self.stem = nn.Sequential(
            nn.Linear(input_dim, stem_dim),
            nn.BatchNorm1d(stem_dim) if batch_norm else nn.Identity(),
            act_cls(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

        blocks = []
        in_dim = stem_dim
        for out_dim in hidden_layer_sizes[1:]:
            blocks.append(ResidualBlock(in_dim, out_dim, act_cls, dropout_rate, batch_norm, se_ratio))
            in_dim = out_dim
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


@Registry.register_method("mlp_residual_classifier")
class MLPResidualMethod(PyTorchBaseMethod):
    def _build_model(self, input_dim, params):
        return PyTorchResidualMLP(
            input_dim=input_dim,
            hidden_layer_sizes=params.get('hidden_layer_sizes', [256, 128, 64]),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.3),
            batch_norm=params.get('batch_norm', True),
            se_ratio=params.get('se_ratio', 0),
        )

    def _get_network_meta(self):
        return {
            "input_dim": self.pytorch_model.stem[0].in_features,
            "hidden_layer_sizes": self.params.get('hidden_layer_sizes', [256, 128, 64]),
            "activation": self.params.get('activation', 'relu'),
            "dropout_rate": self.params.get('dropout_rate', 0.3),
            "batch_norm": self.params.get('batch_norm', True),
            "se_ratio": self.params.get('se_ratio', 0),
        }

    def _build_model_from_meta(self, meta):
        return PyTorchResidualMLP(
            input_dim=meta["input_dim"],
            hidden_layer_sizes=meta["hidden_layer_sizes"],
            activation=meta["activation"],
            dropout_rate=meta["dropout_rate"],
            batch_norm=meta.get("batch_norm", True),
            se_ratio=meta.get("se_ratio", 0),
        )
