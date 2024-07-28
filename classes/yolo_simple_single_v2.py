import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.ReLU()

    def forward(self, x):
        return self.leaky_relu(self.bn(self.conv(x)))


class MyCNN(nn.Module):
    def __init__(self, num_classes=20):
        super(MyCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer4 = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2)
        )
        self.layer5 = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1)
        )
        
        # Parallel convolutions (Inception-like)
        self.parallel_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)
        )
        self.parallel_conv1 = ConvBlock(256, 256, 1, padding=0)
        self.parallel_conv2 = ConvBlock(256, 256, 3, padding=1)
        self.parallel_conv3 = ConvBlock(256, 256, 5, padding=2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(768, 1, 1),
            nn.ReLU()
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        x = self.parallel_conv(x5)
        p1 = self.parallel_conv1(x)
        p2 = self.parallel_conv2(x)
        p3 = self.parallel_conv3(x)
        
        combined = torch.cat([p1, p2, p3], dim=1)
        
        # Attention
        attention_weights = self.attention(combined)
        x = combined * attention_weights
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x