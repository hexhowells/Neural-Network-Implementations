import torch
from torch import nn
import math

from torchvision import models


class GeneratorBlock(nn.Module):
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



class Generator(nn.Module):

	def __init__(self, in_channels=3, out_channels=3):
		super().__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=9, padding=4, bias=False),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.residual_layer = self._make_layers(layers=16, block=GeneratorBlock)

		self.mid_layer = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
			nn.InstanceNorm2d(64, affine=True)
			)

		self.upscale_layer = self._make_layers(layers=2, block=UpscaleBlock)

		self.final_conv = nn.Conv2d(64, 3, kernel_size=9, padding=4, bias=False)
		self.tanh = nn.Tanh()

		# conv layer weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()


	def _make_layers(self, layers, block):
		_layers = []
		for _ in range(layers):
			_layers.append(block())

		return nn.Sequential(*_layers)


	def forward(self, x):
		x = self.layer0(x)
		x_mem = x.clone()

		x = self.residual_layer(x)
		x = self.mid_layer(x)
		x += x_mem

		x = self.upscale_layer(x)
		x = self.final_conv(x)
		x = self.tanh(x)


		return x.to(device='cpu')



class DiscriminatorBlock(nn.Module):

	def __init__(self, in_channels, features, stride):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, features, kernel_size=3, stride=stride, padding=1)
		self.bn = nn.BatchNorm2d(features)
		self.relu = nn.LeakyReLU(0.2, inplace=True)


	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)

		return x



class Discriminator(nn.Module):

	def __init__(self, in_channels=3):
		super().__init__()

		self.layer0 = nn.Sequential(
			nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
			nn.LeakyReLU(0.2, inplace=True)
			)

		self.features = self._make_layers(DiscriminatorBlock, 
			layers=7, 
			features=[64, 128, 128, 256, 256, 512, 512],
			strides=[2, 1, 2, 1, 2, 1, 2])

		self.pool = nn.AdaptiveAvgPool2d(1)

		self.classifier = nn.Sequential(
			nn.Linear(512, 1024),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(1024, 1),
			nn.Sigmoid()
			)


	def _make_layers(self, block, layers, features, strides):
		_layers = []
		prev_features = 64

		for i in range(layers):
			_layers.append(block(prev_features, features[i], strides[i]))
			prev_features = features[i]

		return nn.Sequential(*_layers)
		

	def forward(self, x):
		x = self.layer0(x)
		x = self.features(x)
		x = self.pool(x)

		x = torch.flatten(x, start_dim=1)
		x = self.classifier(x)

		return x.to(device="cpu")




class VGG19(nn.Module):
	def __init__(self):
		super().__init__()

		_vgg = models.vgg19(pretrained=True)
		self.vgg = nn.Sequential(*list(_vgg.features[:36])).eval()

		for param in self.vgg.parameters():
			param.requires_grad = False


	def forward(self, x):
		return self.vgg(x)



class VGG16(nn.Module):
	def __init__(self):
		super().__init__()

		_vgg = models.vgg16(pretrained=True)
		self.vgg = nn.Sequential(*list(_vgg.features[:31])).eval()

		for param in self.vgg.parameters():
			param.requires_grad = False


	def forward(self, x):
		return self.vgg(x)



class ContentLoss(nn.Module):
	def __init__(self, model):
		super().__init__()

		self.model = model
		self.mse = nn.MSELoss()


	def forward(self, sr, hr):
		sr_features = self.model(sr)
		hr_features = self.model(hr)

		content_loss = self.mse(sr_features, hr_features)

		return content_loss
