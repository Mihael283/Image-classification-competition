import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, use_se=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        if use_se:
            layers.append(SEBlock(hidden_dim))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else None

    def forward(self, x):
        out = self.conv(x)
        if self.skip is not None:
            return out + self.skip(x)
        return out

class EnhancedEfficientNet(nn.Module):
    def __init__(self, num_classes=20, width_multiplier=1.0):
        super().__init__()
        
        def round_filters(filters):
            return int(filters * width_multiplier)
        
        self.input_bn = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, round_filters(32), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(round_filters(32))
        self.silu = nn.SiLU(inplace=True)
        
        self.blocks = nn.Sequential(
            MBConvBlock(round_filters(32), round_filters(16), expand_ratio=1, stride=1),
            MBConvBlock(round_filters(16), round_filters(24), expand_ratio=6, stride=2),
            MBConvBlock(round_filters(24), round_filters(24), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(24), round_filters(40), expand_ratio=6, stride=2),
            MBConvBlock(round_filters(40), round_filters(40), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(40), round_filters(80), expand_ratio=6, stride=2),
            MBConvBlock(round_filters(80), round_filters(80), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(80), round_filters(80), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(80), round_filters(112), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(112), round_filters(112), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(112), round_filters(112), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(112), round_filters(192), expand_ratio=6, stride=2),
            MBConvBlock(round_filters(192), round_filters(192), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(192), round_filters(192), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(192), round_filters(192), expand_ratio=6, stride=1),
            MBConvBlock(round_filters(192), round_filters(320), expand_ratio=6, stride=1),
        )
        
        self.conv2 = nn.Conv2d(round_filters(320), round_filters(1280), kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(round_filters(1280))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(round_filters(1280), num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.silu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def MyCNN():
    return EnhancedEfficientNet(width_multiplier=1.0)