import torch
from torch import nn


def conv_bn(in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
	return nn.Sequential(
		nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups),
		nn.BatchNorm2d(out_c)
		)


# Used to upscale the image (height and width) after global average pooling
def scale_dim(x, target_dim):
	_, _, h, w = target_dim
	return x.repeat(1, 1, h, w)


class ShuffleChannels:
	def __init__(self):
		pass

	def __call__(self, x, *args, **kwargs):
		idx = torch.randperm(x.shape[1])
		x = x[:,idx].view(x.size())
		return x


class DownsampleBlock(nn.Module):
	def __init__(self, in_channels, channels, groups=1):
		super().__init__()
		
		self.avg_pool_1 = nn.AvgPool2d(kernel_size=3, stride=(2, 2), padding=1)
		self.conv_bn_1 = conv_bn(in_channels, channels, kernel_size=1, groups=groups)

		self.conv_bn_2 = conv_bn(in_channels, channels, kernel_size=3, stride=2, groups=groups, padding=1)

		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv_3 = nn.Conv2d(in_channels, channels, kernel_size=1, groups=groups)
		self.sigmoid = nn.Sigmoid()

		self.silu = nn.SiLU()


	def forward(self, x):
		x1 = self.avg_pool_1(x)
		x1 = self.conv_bn_1(x1)

		x2 = self.conv_bn_2(x)

		xout = x1 + x2

		x3 = self.global_avg_pool(x)
		x3 = self.conv_3(x3)
		x3 = self.sigmoid(x3)

		x3 = scale_dim(x3, xout.shape)

		xout = torch.matmul(xout, x3)

		xout = self.silu(xout)

		return xout


class FusionBlock(nn.Module):
	def __init__(self, in_channels, channels):
		super().__init__()
		
		self.batch_norm_1 = nn.BatchNorm2d(in_channels)
		self.batch_norm_2 = nn.BatchNorm2d(in_channels)
		self.shuffle = ShuffleChannels()
		self.downsample = DownsampleBlock(in_channels*2, channels, groups=2)


	def forward(self, x1, x2):
		x1 = self.batch_norm_1(x1)
		x2 = self.batch_norm_2(x2)
		x = torch.cat((x1, x2), 1)
		
		x = self.shuffle(x)

		x = self.downsample(x)

		return x


class SSEBlock(nn.Module):
	def __init__(self, in_channels):
		super().__init__()
		self.batch_norm = nn.BatchNorm2d(in_channels)
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		x_bn = self.batch_norm(x)
		x = self.global_avg_pool(x_bn)
		x = self.conv(x)
		x = self.sigmoid(x)

		x = scale_dim(x, x_bn.shape)
		
		x = torch.matmul(x, x_bn)

		return x


class ParNetBlock(nn.Module):
	def __init__(self, in_channels, channels):
		super().__init__()
		
		self.conv_bn_1 = conv_bn(in_channels, channels, kernel_size=1)
		self.conv_bn_2 = conv_bn(in_channels, channels, kernel_size=3, padding=1)

		self.sse = SSEBlock(in_channels)

		self.silu = nn.SiLU()


	def forward(self, x):
		x1 = self.conv_bn_1(x)
		x2 = self.conv_bn_2(x)
		x3 = self.sse(x)

		xout = x1 + x2 + x3
		xout = self.silu(xout)

		return xout


class Stream(nn.Module):
	def __init__(self, channels, length):
		super().__init__()

		blocks = []
		for _ in range(length):
			blocks.append(ParNetBlock(channels, channels))

		self.stream = nn.Sequential(*blocks)

	def forward(self, x):
		return self.stream(x)


class ParNet(nn.Module):
	def __init__(self, in_channels, num_classes, block_c):
		super().__init__()

		self.repvgg_1 = ParNetBlock(in_channels, in_channels)
		self.repvgg_2 = ParNetBlock(in_channels, in_channels)
		self.ds_3 = DownsampleBlock(in_channels, block_c[2])
		self.ds_4 = DownsampleBlock(block_c[2], block_c[3])

		self.stream1 = nn.Sequential(
			Stream(in_channels, 3),
			DownsampleBlock(in_channels, block_c[2])
		)

		self.stream2 = Stream(block_c[2], 4)
		self.fusion1 = FusionBlock(block_c[2], block_c[3])

		self.stream3 = Stream(block_c[3], 4)
		self.fusion2 = FusionBlock(block_c[3], block_c[3])

		self.conv_final = nn.Conv2d(block_c[3], block_c[4], kernel_size=1)
		self.dropout = nn.Dropout(p=0.2)

		self.classifier = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(1, -1),
			nn.Linear(block_c[4], num_classes),
			nn.Dropout(p=0.2),
			nn.LogSoftmax(dim=1)
		)
		

	def forward(self, x):
		x1 = self.repvgg_1(x)
		x1 = self.repvgg_2(x1)
		x2 = self.ds_3(x1)
		x3 = self.ds_4(x2)

		x1 = self.stream1(x1)
		x2 = self.stream2(x2)
		x3 = self.stream3(x3)

		x2 = self.fusion1(x1, x2)
		x3 = self.fusion2(x2, x3)

		x = self.conv_final(x3)
		x = self.dropout(x)

		x = self.classifier(x)
		
		return x.to(device='cpu')



def ParNetSmall(in_channels, num_classes):
	return ParNet(in_channels, num_classes, [64, 96, 192, 384, 1280])


def ParNetMedium(in_channels, num_classes):
	return ParNet(in_channels, num_classes, [64, 128, 256, 512, 2048])


def ParNetLarge(in_channels, num_classes):
	return ParNet(in_channels, num_classes, [64, 160, 320, 640, 2560])


def ParNetExtraLarge(in_channels, num_classes):
	return ParNet(in_channels, num_classes, [64, 200, 400, 800, 3200])