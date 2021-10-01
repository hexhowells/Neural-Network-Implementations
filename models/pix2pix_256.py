import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

torch.set_default_tensor_type("torch.cuda.FloatTensor")


# U-Net
class Generator(nn.Module):
	def __init__(self, in_channels=3, dp=0.5):
		super().__init__()

		# Encoder
		self.encoder_1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
		self.encoder_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.e_batcn_2 = nn.BatchNorm2d(128)
		self.encoder_3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.e_batcn_3 = nn.BatchNorm2d(256)
		self.encoder_4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
		self.e_batcn_4 = nn.BatchNorm2d(512)
		self.encoder_5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.e_batcn_5 = nn.BatchNorm2d(512)
		self.encoder_6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.e_batcn_6 = nn.BatchNorm2d(512)
		self.encoder_7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.e_batcn_7 = nn.BatchNorm2d(512)

		# Decoder
		self.decoder_1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
		self.dropout_1 = nn.Dropout(p=dp)
		self.decoder_2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
		self.dropout_2 = nn.Dropout(p=dp)
		self.decoder_3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
		self.dropout_3 = nn.Dropout(p=dp)
		self.decoder_4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1)
		self.dropout_4 = nn.Dropout(p=dp)
		self.decoder_5 = nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1)
		self.dropout_5 = nn.Dropout(p=dp)
		self.decoder_6 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
		self.dropout_6 = nn.Dropout(p=dp)
		self.decoder_7 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)


	def forward(self, x):

		x_e1 = F.leaky_relu(self.encoder_1(x), negative_slope=0.2)
		x_e2 = F.leaky_relu(self.e_batcn_2(self.encoder_2(x_e1)), negative_slope=0.2)
		x_e3 = F.leaky_relu(self.e_batcn_3(self.encoder_3(x_e2)), negative_slope=0.2)
		x_e4 = F.leaky_relu(self.e_batcn_4(self.encoder_4(x_e3)), negative_slope=0.2)
		x_e5 = F.leaky_relu(self.e_batcn_5(self.encoder_5(x_e4)), negative_slope=0.2)
		x_e6 = F.leaky_relu(self.e_batcn_6(self.encoder_6(x_e5)), negative_slope=0.2)
		x_e7 = F.leaky_relu(self.e_batcn_7(self.encoder_7(x_e6)), negative_slope=0.2)

		x_d7 = F.relu(self.dropout_1(self.decoder_1(x_e7)))
		x_d7 = torch.cat((x_e6, x_d7), dim=1)
		x_d6 = F.relu(self.dropout_2(self.decoder_2(x_d7)))
		x_d6 = torch.cat((x_e5, x_d6), dim=1)
		x_d5 = F.relu(self.dropout_3(self.decoder_3(x_d6)))
		x_d5 = torch.cat((x_e4, x_d5), dim=1)
		x_d4 = F.relu(self.dropout_4(self.decoder_4(x_d5)))
		x_d4 = torch.cat((x_e3, x_d4), dim=1)
		x_d3 = F.relu(self.dropout_5(self.decoder_5(x_d4)))
		x_d3 = torch.cat((x_e2, x_d3), dim=1)
		x_d2 = F.relu(self.dropout_6(self.decoder_6(x_d3)))
		x_d2 = torch.cat((x_e1, x_d2), dim=1)
		x_d1 = torch.tanh(self.decoder_7(x_d2))

		return x_d1.to(device='cpu')



# PatchGAN 70x70
class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv_1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
		self.conv_2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
		self.batc_2 = nn.BatchNorm2d(128)
		self.conv_3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
		self.batc_3 = nn.BatchNorm2d(256)
		self.conv_4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1)
		self.batc_4 = nn.BatchNorm2d(512)
		self.conv_5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)


	def forward(self, x):

		x = F.leaky_relu(self.conv_1(x), negative_slope=0.2)
		x = F.leaky_relu(self.batc_2(self.conv_2(x)), negative_slope=0.2)
		x = F.leaky_relu(self.batc_3(self.conv_3(x)), negative_slope=0.2)
		x = F.leaky_relu(self.batc_4(self.conv_4(x)), negative_slope=0.2)
		x = torch.sigmoid(self.conv_5(x))
		x = torch.mean(x, (1,2,3))
		x = torch.transpose(x.reshape(1,-1), 0, 1)

		return x.to(device='cpu')