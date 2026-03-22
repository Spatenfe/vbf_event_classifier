from ml_framework.core.registry import Registry
from ml_framework.methods.pytorch_base import PyTorchBaseMethod, SEBlock
import torch.nn as nn


_ACT_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(100,), activation='relu',
                 dropout_rate=0.2, batch_norm=False, se_ratio=0):
        super().__init__()
        layers = []
        in_dim = input_dim
        act_cls = _ACT_MAP.get(activation, nn.ReLU)

        for h_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            if se_ratio > 0:
                layers.append(SEBlock(h_dim, ratio=se_ratio))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


@Registry.register_method("mlp_classifier")
class MLPMethod(PyTorchBaseMethod):
    def _build_model(self, input_dim, params):
        return PyTorchMLP(
            input_dim=input_dim,
            hidden_layer_sizes=params.get('hidden_layer_sizes', [100]),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.45),
            batch_norm=params.get('batch_norm', False),
            se_ratio=params.get('se_ratio', 0),
        )

    def _get_network_meta(self):
        return {
            "input_dim": self.pytorch_model.network[0].in_features,
            "hidden_layer_sizes": self.params.get('hidden_layer_sizes', [100]),
            "activation": self.params.get('activation', 'relu'),
            "dropout_rate": self.params.get('dropout_rate', 0.45),
            "batch_norm": self.params.get('batch_norm', False),
            "se_ratio": self.params.get('se_ratio', 0),
        }

    def _build_model_from_meta(self, meta):
        return PyTorchMLP(
            input_dim=meta["input_dim"],
            hidden_layer_sizes=meta["hidden_layer_sizes"],
            activation=meta["activation"],
            dropout_rate=meta["dropout_rate"],
            batch_norm=meta.get("batch_norm", False),
            se_ratio=meta.get("se_ratio", 0),
        )
