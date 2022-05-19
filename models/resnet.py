import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class Bottleneck(nn.Module):
	expansion = 4
	def __init__(self, in_channels, channels, stride=1):
		super(Bottleneck, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.conv3 = nn.Conv2d(channels, self.expansion*channels, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*channels)

		if stride != 1 or in_channels != self.expansion*channels:
			self.projection = nn.Sequential(
				nn.Conv2d(in_channels, self.expansion*channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*channels)
			)
		else:
			self.projection = nn.Identity()


	def forward(self, x):
		x_res = x.clone()

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x += self.projection(x_res)
		x = F.relu(x)

		return x


class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_channels, channels, stride=1):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)

		if stride != 1 or in_channels != channels:
			self.projection = nn.Sequential(
				nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(channels)
			)
		else:
			self.projection = nn.Identity()


	def forward(self, x):
		x_res = x.clone()

		x = F.relu(self.bn1(self.conv1(x)))
		x = self.bn2(self.conv2(x))
		x += self.projection(x_res)
		x = F.relu(x)

		return x


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes):
		super(ResNet, self).__init__()
		self.in_channels = 64

		self.layer0 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(3, stride=2),
			nn.ReLU()
		)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1])
		self.layer3 = self._make_layer(block, 256, layers[2])
		self.layer4 = self._make_layer(block, 512, layers[3])
		self.global_pool = nn.AdaptiveAvgPool2d((1,1))
		self.linear = nn.Linear(512*block.expansion, num_classes)


	def _make_layer(self, block, channels, layers, stride=1):
		strides = [stride] + [1]*(layers-1)
		blocks = []
		for stride in strides:
			blocks.append(block(self.in_channels, channels, stride))
			self.in_channels = channels * block.expansion

		return nn.Sequential(*blocks)


	def forward(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.global_pool(x)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		return x.to(device='cpu')


def ResNet18(num_classes=10):
	return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=10):
	return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)