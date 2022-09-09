from SRGAN import VGG16
import torch
from torch import nn



class ContentLoss(nn.Module):
	def __init__(self, model):
		super().__init__()

		self.model = model
		self.lossfn = nn.MSELoss()


	def forward(self, sr_images, hr_images):
		sr_out = self.model(sr_images)
		hr_out = self.model(hr_images)

		content_loss = self.lossfn(sr_out, hr_out)

		return content_loss