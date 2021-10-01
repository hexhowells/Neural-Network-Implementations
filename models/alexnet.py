import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

torch.set_default_tensor_type("torch.cuda.FloatTensor")

class AlexNet1(torch.nn.Module):
	def __init__(self, n_class):
		super(AlexNet, self).__init__()

		self.conv1 = nn.Conv2d(3, 96, kernel_size=9, stride=3)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

		self.conv2 = nn.Conv2d(96, 256, kernel_size=5)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

		self.conv3 = nn.Conv2d(256, 128, kernel_size=5)

		#self.con4 = nn.Conv2d(384, 384, kernel_size=3)

		self.conv5 = nn.Conv2d(128, 64, kernel_size=3)
		self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

		self.linear1 = nn.Linear(576, 2048)
		self.dropout1 = nn.Dropout(p=0.5)
		self.linear2 = nn.Linear(2048, 2048)
		self.dropout2 = nn.Dropout(p=0.5)
		self.linear3 = nn.Linear(2048, n_class)


	def forward(self, x):

		x = F.relu(self.conv1(x))
		x = F.relu(self.pool1(x))

		x = F.relu(self.conv2(x))
		x = F.relu(self.pool2(x))

		x = F.relu(self.conv3(x))

		#x = F.relu(self.con4(x))

		x = F.relu(self.conv5(x))
		x = F.relu(self.pool5(x))

		x = x.view(-1, 576)
		x = F.relu(self.linear1(x))
		x = F.relu(self.dropout1(x))
		x = F.relu(self.linear2(x))
		x = F.relu(self.dropout2(x))
		x = self.linear3(x)

		return x.to(device='cpu')


class AlexNet(nn.Module):
    def __init__(self, n_class):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=8, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_class),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x.to(device='cpu')


