import torch
from torch import nn


class SRCNN(nn.Module):

	def __init__(self, in_channels=3, out_channels=3):
		super().__init__()

		self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(32, out_channels, kernel_size=5, padding=2)
		self.relu = nn.ReLU(inplace=True)


	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.relu(self.conv2(x))
		x = self.conv3(x)

		return x.to(device='cpu')