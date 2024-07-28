import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Union, Optional

class SimpleNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 20,
        in_chans: int = 1,
        drop_rates: Dict[int, float] = {},
    ):
        super(SimpleNet, self).__init__()
        self.cfg: List[Tuple[Union[int, str], int, Union[float, None], Optional[str]]] = [
            (64, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (128, 1, None),
            ("p", 2, 0.0),
            (128, 1, 0.0),
            (128, 1, 0.0),
            (256, 1, None),
            ("p", 2, 0.0),
            (256, 1, 0.0),
            (256, 1, None),
            ("p", 2, 0.0),
            (512, 1, 0.0),
            (1024, 1, 0.0, "k1"),
            (256, 1, None, "k1"),
        ]
        self.dropout_rates = {int(key): float(value) for key, value in drop_rates.items()}
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)

        self.features = self._make_layers(in_chans)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.global_pool(out)
        out = F.dropout(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, in_chans):
        layers: List[nn.Module] = []
        input_channel = in_chans
        for idx, (layer, stride, default_dropout_rate, *layer_type) in enumerate(self.cfg):
            custom_dropout = self.dropout_rates.get(idx, default_dropout_rate)
            custom_dropout = None if custom_dropout is None else float(custom_dropout)
            kernel_size = 1 if layer_type == ['k1'] else 3
            padding = 0 if layer_type == ['k1'] else 1

            if layer == "p":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(input_channel, layer, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(layer),
                    nn.ReLU(inplace=True),
                ]
                if custom_dropout is not None:
                    layers.append(nn.Dropout2d(p=custom_dropout, inplace=False))
                input_channel = layer

        return nn.Sequential(*layers)

def MyCNN(num_classes: int = 20, in_chans: int = 1, **kwargs: Any) -> SimpleNet:
    return SimpleNet(num_classes=num_classes, in_chans=in_chans, **kwargs)
