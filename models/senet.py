import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class BasicBlock(nn.Module):
	expansion = 1
	ratio = 16
	def __init__(self, in_channels, channels, stride=1):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)

		self.se_pool = nn.AdaptiveAvgPool2d(1)
		self.se_linear1 = nn.Linear(channels, channels//self.ratio, bias=False)
		self.se_linear2 = nn.Linear(channels//self.ratio, channels, bias=False)

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
		b, c, _, _ = x.size()

		# SE Block
		se_x = self.se_pool(x).view(b, c)
		se_x = F.relu(self.se_linear1(se_x))
		se_x = torch.sigmoid(self.se_linear2(se_x))
		se_x = se_x.view(b, c, 1, 1)
		x *= se_x.expand_as(x)

		x += self.projection(x_res)
		x = F.relu(x)

		return x


class SEResNet(nn.Module):
	def __init__(self, block, layers, num_classes):
		super(SEResNet, self).__init__()
		self.in_channels = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.linear = nn.Linear(512, num_classes)


	def _make_layer(self, block, channels, layers, stride=1):
		strides = [stride] + [1]*(layers-1)
		blocks = []
		for stride in strides:
			blocks.append(block(self.in_channels, channels, stride))
			self.in_channels = channels * block.expansion

		return nn.Sequential(*blocks)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = F.avg_pool2d(x, 4)
		x = x.view(x.size(0), -1)
		x = self.linear(x)
		return x.to(device='cpu')


def SEResNet18(num_classes=10):
	return SEResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def SEResNet34(num_classes=10):
	return SEResNet(BasicBlock, [3, 4, 6, 3], num_classes)