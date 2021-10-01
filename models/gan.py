import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

torch.set_default_tensor_type("torch.cuda.FloatTensor")


class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		self.linear_1 = nn.Linear(4*4, 128)
		self.linear_2 = nn.Linear(128, 28*28)


	def forward(self, x):

		x = F.relu(self.linear_1(x))
		x = torch.tanh(self.linear_2(x))

		x = torch.reshape(x, (-1, 28, 28))

		return x.to(device='cpu')


class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.linear_1 = nn.Linear(28*28, 128)
		self.linear_2 = nn.Linear(128, 1)


	def forward(self, x):

		x = torch.flatten(x, start_dim=1, end_dim=-1)
		x = F.relu(self.linear_1(x))
		x = torch.sigmoid(self.linear_2(x))

		return x.to(device='cpu')



