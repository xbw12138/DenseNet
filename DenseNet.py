import math
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchsummary import summary


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        y = self.conv1(func.relu(self.bn1(x)))
        y = self.conv2(func.relu(self.bn2(y)))
        x = torch.cat([y, x], 1)
        return x


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(func.relu(self.bn(x)))
        x = func.avg_pool2d(x, 2)
        return x


class DenseNet(nn.Module):
    def __init__(self, block, num_block, growth_rate=12, reduction=0.5, num_classes=10, fn_size=1, pool_size=7):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.pool_size = pool_size

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes * fn_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(1, x.size())
        x = self.conv1(x)
        # print(2, x.size())
        x = self.pool1(x)
        # print(3, x.size())
        x = self.trans1(self.dense1(x))
        # print(4, x.size())
        x = self.trans2(self.dense2(x))
        # print(5, x.size())
        x = self.trans3(self.dense3(x))
        # print(6, x.size())
        x = self.dense4(x)
        # print(7, x.size())
        x = func.avg_pool2d(func.relu(self.bn(x)), self.pool_size)
        # print(8, x.size())
        x = x.view(x.size(0), -1)
        # print(9, x.size())
        x = self.linear(x)
        x = nn.Softmax(1)(x)
        # print(10, x.size())
        return x


def DenseNet121(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32, num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet169(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32, num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet201(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32, num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


def DenseNet161(fn_size=1, pool_size=7, num_classes=4):
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48, num_classes=num_classes, fn_size=fn_size, pool_size=pool_size)


if __name__ == "__main__":
    # size, fn_size, pool_size = 1024, 16, 8
    # size, fn_size, pool_size = 512, 4, 8
    # size, fn_size, pool_size = 256, 1, 8
    size, fn_size, pool_size = 256, 4, 4
    test_input = torch.rand(1, 3, size, size)
    model = DenseNet121(fn_size, pool_size, 4)
    summary(model, (3, size, size))
    output = model(test_input)
    print(output)