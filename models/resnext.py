import torch
import torch.nn as nn
import torch.nn.functional as F

class BottlenBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, path=32,stride=1):
        super(BottlenBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=path,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.pass_by = nn.Sequential()

        if stride != 1 or in_planes != planes * self.expansion:
            self.pass_by = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.pass_by(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, path,num_class=10):
        super(ResNeXt, self).__init__()

        self.inplanes = 64
        self.path = path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0],  path=self.path)
        self.layer2 = self._make_layer(block, 128, layers[1], path=self.path,stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], path=self.path,stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], path=self.path,stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, plane, layer, path, stride=1):
        layers = []
        layers.append(block(self.inplanes, plane, path,stride))
        self.inplanes = plane * block.expansion
        for i in range(1, layer):
            layers.append(block(self.inplanes, plane,path))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNeXt50_32x4d(num_class=10):
    model = ResNeXt(BottlenBlock, [3, 4, 6, 3], 32,num_class)
    return model


def test():
    net = ResNeXt50_32x4d()
    count = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            count += 1

    print(count)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.shape)


if __name__ == "__main__":
    test()

