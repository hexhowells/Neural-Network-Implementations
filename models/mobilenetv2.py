import torch
from torch import nn
import torch.nn.functional as F

torch.set_default_tensor_type("torch.cuda.FloatTensor")


# function taken from original paper
def _make_divisible(v, divisor, min_value=None):
	if min_value is None:
		min_value = divisor

	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor

	return new_v


def convBnReLU(in_c, out_c, kernel_size, stride, padding):
	return nn.Sequential(
		nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
		nn.BatchNorm2d(out_c), 
		nn.ReLU6(inplace=True))



class InvertedResidual(nn.Module):
	def __init__(self, in_c, out_c, stride, expansion):
		super(InvertedResidual, self).__init__()
		self.identity = (stride == 1) and (in_c == out_c)
		hidden_dim = round(expansion * in_c)

		if expansion == 1:
			self.conv = nn.Sequential(
				nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),

				nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_c)
				)
		else:
			self.conv = nn.Sequential(
				nn.Conv2d(in_c, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),

				nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),

				nn.Conv2d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_c)
				)


	def forward(self, x):
		if self.identity:
			return x + self.conv(x)
		else:
			return self.conv(x)



class MobileNetV2(nn.Module):
	def __init__(self, num_classes, width_mult=1.0):
		super(MobileNetV2, self).__init__()

		self.configs = [
			# t, c, n, s
			[1,  16, 1, 1],
			[6,  24, 2, 2],
			[6,  32, 3, 2],
			[6,  64, 4, 2],
			[6,  96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		# first conv layer
		self.conv1 = convBnReLU(3, 32, kernel_size=3, stride=2, padding=1)

		# bottleneck / mobilenet layers
		in_c = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
		layers = []

		for t, c, n, s in self.configs:
			out_c = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
			for i in range(n):
				stride = s if i == 0 else 1
				layers.append(InvertedResidual(in_c, out_c, stride, expansion=t))
				in_c = out_c

		self.mobile = nn.Sequential(*layers)

		# output conv layer
		self.conv2 = convBnReLU(320, 1280, kernel_size=1, stride=1, padding=1)

		# classifier
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1280, num_classes)
			)
		

	def forward(self, x):
		
		x = self.conv1(x)

		x = self.mobile(x)

		x = self.conv2(x)
		x = self.avgpool(x)
		x = torch.flatten(x, start_dim=1, end_dim=-1)
		x = self.classifier(x)

		return x.to(device='cpu')
