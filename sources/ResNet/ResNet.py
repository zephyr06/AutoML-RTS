import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != self.expansion * out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion * out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        # out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, prune_ratio=0, quant_type="qint8", num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.quant_type = quant_type
        self.prune_ratio = prune_ratio
        # TODO: add prune and quantization there!
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        if (self.quant_type == "qint8"):
            self.conv1 = quantization.quantize_dynamic(self.conv1,
                                                       {nn.Conv2d,
                                                           nn.BatchNorm2d, nn.ReLU},
                                                       dtype=torch.qint8)  # float16  or qint8
        elif (self.quant_type == "float16"):
            self.conv1 = quantization.quantize_dynamic(self.conv1,
                                                       {nn.Conv2d,
                                                           nn.BatchNorm2d, nn.ReLU},
                                                       dtype=torch.float16)  # float16  or qint8
        else:  # no quantization
            pass
        self.conv1[0] = prune.random_unstructured(
            self.conv1[0], name="weight", amount=self.prune_ratio)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []

        block_obj = block(self.inplanes, planes, stride, downsample)
        if (self.quant_type == "qint8"):
            block_obj = quantization.quantize_dynamic(block_obj, {nn.Conv2d, nn.BatchNorm2d, nn.ReLU},
                                                      dtype=torch.qint8)  # float16  or qint8
        elif (self.quant_type == "float16"):
            block_obj = quantization.quantize_dynamic(block_obj, {nn.Conv2d, nn.BatchNorm2d, nn.ReLU},
                                                      dtype=torch.float16)  # float16  or qint8
        else:  # no quantization
            pass
        block_obj.conv1[0] = prune.random_unstructured(
            block_obj.conv1[0], name="weight", amount=self.prune_ratio)
        block_obj.conv2[0] = prune.random_unstructured(
            block_obj.conv2[0], name="weight", amount=self.prune_ratio)
        layers.append(block_obj)
        self.inplanes = planes
        for i in range(1, blocks):
            block_obj = block(self.inplanes, planes)
            if (self.quant_type == "qint8"):
                block_obj = quantization.quantize_dynamic(block_obj, {nn.Conv2d, nn.BatchNorm2d, nn.ReLU},
                                                          dtype=torch.qint8)  # float16  or qint8
            elif (self.quant_type == "float16"):
                block_obj = quantization.quantize_dynamic(block_obj, {nn.Conv2d, nn.BatchNorm2d, nn.ReLU},
                                                          dtype=torch.float16)  # float16  or qint8
            else:  # no quantization
                pass
            block_obj.conv1[0] = prune.random_unstructured(
                block_obj.conv1[0], name="weight", amount=self.prune_ratio)
            block_obj.conv2[0] = prune.random_unstructured(
                block_obj.conv2[0], name="weight", amount=self.prune_ratio)
            # layers.append(block(self.inplanes, planes))
            layers.append(block_obj)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def get_resnet_blocks(num_layers):
    # Dictionary mapping the total number of layers to the number of layer blocks in each stage
    resnet_configs = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        # Add more configurations as needed
    }

    # Check if the provided number of layers is in the dictionary
    if num_layers in resnet_configs:
        return resnet_configs[num_layers]
    else:
        # Redistribute layers for other cases
        base_blocks = [3, 4, 6, 3]  # Base configuration for redistribution
        total_blocks = sum(base_blocks)
        redistributed_blocks = [
            int(round(b * (num_layers - 2) / total_blocks)) for b in base_blocks]

        # Adjust to ensure the total number of layers is exactly num_layers
        diff = num_layers - 2 - sum(redistributed_blocks)
        # Add the difference to the first stage
        redistributed_blocks[0] += diff

        return redistributed_blocks
