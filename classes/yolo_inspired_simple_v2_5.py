import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Mish()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = ConvBlock(in_channels, in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(MyCNN, self).__init__()
        
        self.initial_conv = ConvBlock(1, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4)
        self.layer3 = self._make_layer(256, 512, 6)
        self.layer4 = self._make_layer(512, 1024, 3)
        
        self.se_block = SEBlock(1024)
        
        self.parallel_conv = ConvBlock(1024, 512, kernel_size=1, padding=0)
        self.parallel_conv1 = ConvBlock(512, 512, kernel_size=1, padding=0)
        self.parallel_conv2 = ConvBlock(512, 512, kernel_size=3, padding=1)
        self.parallel_conv3 = ConvBlock(512, 512, kernel_size=5, padding=2)
        
        self.attention = nn.Sequential(
            nn.Conv2d(1536, 1, 1),
            nn.ReLU()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(1536, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ConvBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.se_block(x)
        
        x = self.parallel_conv(x)
        p1 = self.parallel_conv1(x)
        p2 = self.parallel_conv2(x)
        p3 = self.parallel_conv3(x)
        
        combined = torch.cat([p1, p2, p3], dim=1)
        
        attention_weights = self.attention(combined)
        x = combined * attention_weights
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

