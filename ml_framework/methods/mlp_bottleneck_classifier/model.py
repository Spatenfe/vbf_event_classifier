from ml_framework.core.registry import Registry
from ml_framework.methods.pytorch_base import PyTorchBaseMethod
from ml_framework.methods.mlp_classifier.model import _ACT_MAP
import torch
import torch.nn as nn


class PyTorchBottleneckMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(256, 128, 64), activation='relu',
                 dropout_rate=0.45, batch_norm=False):
        super().__init__()
        act_cls = _ACT_MAP.get(activation, nn.ReLU)
        self.act = act_cls()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.down_layers = nn.ModuleList()
        self.down_bn = nn.ModuleList()
        in_dim = input_dim
        for h_dim in hidden_layer_sizes:
            self.down_layers.append(nn.Linear(in_dim, h_dim))
            self.down_bn.append(nn.BatchNorm1d(h_dim) if batch_norm else nn.Identity())
            in_dim = h_dim

        bottleneck_dim = max(8, in_dim // 2)
        self.bottleneck = nn.Linear(in_dim, bottleneck_dim)
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim) if batch_norm else nn.Identity()

        self.up_layers = nn.ModuleList()
        self.up_bn = nn.ModuleList()
        current_in_dim = bottleneck_dim
        for i in range(len(hidden_layer_sizes) - 1, -1, -1):
            skip_dim = hidden_layer_sizes[i]
            out_dim = hidden_layer_sizes[i - 1] if i > 0 else hidden_layer_sizes[0]
            self.up_layers.append(nn.Linear(current_in_dim + skip_dim, out_dim))
            self.up_bn.append(nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity())
            current_in_dim = out_dim

        self.output_layer = nn.Linear(current_in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        current_x = x
        for layer, bn in zip(self.down_layers, self.down_bn):
            current_x = self.dropout(self.act(bn(layer(current_x))))
            skips.append(current_x)

        x = self.dropout(self.act(self.bottleneck_bn(self.bottleneck(current_x))))

        for i, (layer, bn) in enumerate(zip(self.up_layers, self.up_bn)):
            x = torch.cat([x, skips[-(i + 1)]], dim=1)
            x = self.dropout(self.act(bn(layer(x))))

        return self.sigmoid(self.output_layer(x))


@Registry.register_method("mlp_bottleneck_classifier")
class MLPBottleneckMethod(PyTorchBaseMethod):
    def _build_model(self, input_dim, params):
        return PyTorchBottleneckMLP(
            input_dim=input_dim,
            hidden_layer_sizes=params.get('hidden_layer_sizes', [256, 128, 64]),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.45),
            batch_norm=params.get('batch_norm', False),
        )

    def _get_network_meta(self):
        return {
            "input_dim": self.pytorch_model.down_layers[0].in_features,
            "hidden_layer_sizes": self.params.get('hidden_layer_sizes', [256, 128, 64]),
            "activation": self.params.get('activation', 'relu'),
            "dropout_rate": self.params.get('dropout_rate', 0.45),
            "batch_norm": self.params.get('batch_norm', False),
        }

    def _build_model_from_meta(self, meta):
        return PyTorchBottleneckMLP(
            input_dim=meta["input_dim"],
            hidden_layer_sizes=meta["hidden_layer_sizes"],
            activation=meta["activation"],
            dropout_rate=meta["dropout_rate"],
            batch_norm=meta.get("batch_norm", False),
        )
