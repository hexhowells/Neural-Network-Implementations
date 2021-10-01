import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class DepthwiseSeparable(nn.Module):
	def __init__(self, in_c, out_c, stride):
		super(DepthwiseSeparable, self).__init__()

		self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, padding=1, groups=in_c, bias=False)
		self.bn1 = nn.BatchNorm2d(in_c)
		self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_c)


	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		return x


class MobileNetV1(nn.Module):
	def __init__(self, num_classes):
		super(MobileNetV1, self).__init__()

		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32)
		self.mobilelayers = nn.Sequential(
			DepthwiseSeparable(32, 64, 1),
			DepthwiseSeparable(64, 128, 2),
			DepthwiseSeparable(128, 128, 1),
			DepthwiseSeparable(128, 256, 2),
			DepthwiseSeparable(256, 256, 1),
			DepthwiseSeparable(256, 512, 2),

			DepthwiseSeparable(512, 512, 1),
			DepthwiseSeparable(512, 512, 1),
			DepthwiseSeparable(512, 512, 1),
			DepthwiseSeparable(512, 512, 1),
			DepthwiseSeparable(512, 512, 1),

			DepthwiseSeparable(512, 1024, 2),
			DepthwiseSeparable(1024, 1024, 2)
			)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.linear = nn.Linear(1024, num_classes)
		

	def forward(self, x):

		x = F.relu(self.bn1(self.conv1(x)))
		x = self.mobilelayers(x)
		x = self.avgpool(x)
		x = torch.flatten(x, start_dim=1, end_dim=-1)
		x = self.linear(x)

		return x.to(device='cpu')
