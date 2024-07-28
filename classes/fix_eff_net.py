import torch
import torch.nn as nn

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        
        self.conv = nn.Sequential(
            # Expansion phase
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            # Projection phase
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection if in and out dimensions are the same (Modified)
        self.skip = None
        if stride == 1 and in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        if self.skip is not None:
            return self.conv(x) + self.skip(x)
        return self.conv(x)

class FixEfficientNet(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        
        # Change input channels to 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.silu = nn.SiLU()
        
        # MBConv blocks (no change needed here)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1, stride=1),
            MBConvBlock(16, 24, expand_ratio=6, stride=2),
            MBConvBlock(24, 24, expand_ratio=6, stride=1),
            MBConvBlock(24, 40, expand_ratio=6, stride=2),
            MBConvBlock(40, 40, expand_ratio=6, stride=1),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
            MBConvBlock(80, 80, expand_ratio=6, stride=1),
            MBConvBlock(80, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 112, expand_ratio=6, stride=1),
            MBConvBlock(112, 192, expand_ratio=6, stride=2),
            MBConvBlock(192, 192, expand_ratio=6, stride=1),
            MBConvBlock(192, 320, expand_ratio=6, stride=1),
        )
        
        # Final conv layer (no change needed)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Classifier (no change needed)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        
        # Change input batch norm to 1 channel
        self.input_bn = nn.BatchNorm2d(1)

    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        # Apply input batch normalization
        x = self.input_bn(input_images)
        
        # Initial conv layer
        x = self.silu(self.bn1(self.conv1(x)))
        
        # MBConv blocks
        x = self.blocks(x)
        
        # Final conv layer
        x = self.silu(self.bn2(self.conv2(x)))
        
        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def MyCNN():
    return FixEfficientNet()