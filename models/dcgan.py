import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv_1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1)
		self.batc_1 = nn.BatchNorm2d(512)
		self.conv_2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
		self.batc_2 = nn.BatchNorm2d(256)
		self.conv_3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
		self.batc_3 = nn.BatchNorm2d(128)
		self.conv_4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
		self.batc_4 = nn.BatchNorm2d(64)
		self.conv_5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)


	def forward(self, x):

		x = F.relu(self.batc_1(self.conv_1(x)))
		x = F.relu(self.batc_2(self.conv_2(x)))
		x = F.relu(self.batc_3(self.conv_3(x)))
		x = F.relu(self.batc_4(self.conv_4(x)))
		x = torch.tanh(self.conv_5(x))

		x = torch.reshape(x, (-1, 3, 64, 64))

		return x.to(device='cpu')


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv_1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
		self.conv_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.batc_2 = nn.BatchNorm2d(128)
		self.conv_3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.batc_3 = nn.BatchNorm2d(256)
		self.conv_4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.batc_4 = nn.BatchNorm2d(512)
		self.conv_5 = nn.Conv2d(512, 1, kernel_size=4, stride=1)
		


	def forward(self, x):

		x = F.leaky_relu(self.conv_1(x), negative_slope=0.2)
		x = F.leaky_relu(self.batc_2(self.conv_2(x)), negative_slope=0.2)
		x = F.leaky_relu(self.batc_3(self.conv_3(x)), negative_slope=0.2)
		x = F.leaky_relu(self.batc_4(self.conv_4(x)), negative_slope=0.2)
		x = torch.sigmoid(self.conv_5(x))
		x = torch.reshape(x, (-1, 1))
		

		return x.to(device='cpu')



