import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        reduction = 0.5
        if stride == 2:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25

        # Define the layers
        self.conv1 = nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv2 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(int(in_channels * reduction * 0.5))
        self.conv3 = nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv4 = nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=True)
        self.bn4 = nn.BatchNorm2d(int(in_channels * reduction))
        self.conv5 = nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=True)
        self.bn5 = nn.BatchNorm2d(out_channels)

        # Shortcut to match dimensions and channels
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = self.bn5(self.conv5(output))  # Remove ReLU activation here
        shortcut = self.shortcut(input)
        output += shortcut  # Remove ReLU activation on the shortcut
        output = F.relu(output)  # Apply ReLU activation after adding the shortcut
        return output

class SqueezeNext(nn.Module):
    def __init__(self, width_x, blocks, num_classes):
        super(SqueezeNext, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, int(width_x * self.in_channels), 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(width_x * self.in_channels))
        self.stage1 = self._make_layer(blocks[0], width_x, 32, 1)
        self.stage2 = self._make_layer(blocks[1], width_x, 64, 2)
        self.stage3 = self._make_layer(blocks[2], width_x, 128, 2)
        self.stage4 = self._make_layer(blocks[3], width_x, 256, 2)
        self.conv2 = nn.Conv2d(int(width_x * self.in_channels), int(width_x * 128), 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(width_x * 128))

        # Calculate the output size after the convolutional layers
        self.output_size = self._calculate_output_size(width_x)
        self.linear = nn.Linear(self.output_size, num_classes)

    def _make_layer(self, num_block, width_x, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for _stride in strides:
            layers.append(BasicBlock(int(width_x * self.in_channels), int(width_x * out_channels), _stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _calculate_output_size(self, width_x):
        # Calculate the output size after the convolutional layers
        sample_input = torch.randn(1, 1, 100, 100)
        sample_output = self.forward_features(sample_input)
        output_size = sample_output.view(-1).size(0)
        return output_size

    def forward_features(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.adaptive_avg_pool2d(output, (1, 1))
        return output

    def forward(self, input):
        output = self.forward_features(input)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

def MyCNN():
    return SqueezeNext(2.0, [2, 4, 14, 1], 20)