import torch
from torch import nn
import math


class BasicBlock(nn.Module):
	def __init__(self, channels=64):
		super().__init__()

		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(channels)
		self.relu = nn.LeakyReLU(0.2, inplace=True)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(channels)
		self.in2 = nn.InstanceNorm2d(channels, affine=True)


	def forward(self, x):
		x_mem = x.clone()

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)
		x = self.in2(x)

		x += x_mem

		return x



class UpscaleBlock(nn.Module):
	def __init__(self, in_channels=64, channels=256):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
		self.pixel_shuffle = nn.PixelShuffle(2)
		self.relu = nn.LeakyReLU(0.2, inplace=True)


	def forward(self, x):
		x = self.conv(x)
		x = self.pixel_shuffle(x)
		x = self.relu(x)


		return x



class SRResNet(nn.Module):

	def __init__(self, in_channels=3, out_channels=3):
		super().__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=9, padding=4, bias=False),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.residual_layer = self._make_layers(num_layers=16, block=BasicBlock)

		self.mid_layer = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm2d(64, affine=True)
			)

		self.upscale_layer = self._make_layers(num_layers=2, block=UpscaleBlock)

		self.final_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=False)

		# conv layer weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()


	def _make_layers(self, num_layers, block):
		layers = []
		for _ in range(num_layers):
			layers.append(block())

		return nn.Sequential(*layers)


	def forward(self, x):
		x = self.layer0(x)
		x_mem = x.clone()

		x = self.residual_layer(x)
		x = self.mid_layer(x)
		x += x_mem

		x = self.upscale_layer(x)
		x = self.final_conv(x)


		return x.to(device='cpu')