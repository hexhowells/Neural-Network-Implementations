import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class SuperResolution(torch.utils.data.Dataset):

	def __init__(self, root_dir, transform=None, crop=True):
		self.root_dir = root_dir
		if self.root_dir[-1] != '/':
			self.root_dir += '/'

		
		self.transform = transform
		self.lr_files = []
		self.hr_files = []

		
		for file in os.listdir(self.root_dir + "LR"):
			self.lr_files.append("LR" + "/" + file)

		for file in os.listdir(self.root_dir + "HR"):
			self.hr_files.append("HR" + "/" + file)

		self.size = len(self.lr_files)

		self.crop = crop


	def __len__(self):
		return self.size


	def __getitem__(self, index):
		# load low res file
		lr_filepath = self.root_dir + self.lr_files[index]

		lr_image = cv2.imread(lr_filepath)
		lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
		lr_image = np.moveaxis(lr_image, -1, 0)
		x = torch.from_numpy(lr_image / 127.5 - 1.0)


		# load high res file
		hr_filepath = self.root_dir + self.hr_files[index]

		hr_image = cv2.imread(hr_filepath)
		hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
		hr_image = np.moveaxis(hr_image, -1, 0)
		y = torch.from_numpy(hr_image / 127.5 - 1.0)
		
		if self.transform:
			x = self.transform(x)
			y = self.transform(y)

		if self.crop:
			i, j, h, w = transforms.RandomCrop.get_params(x, output_size=(33, 33))

			x = TF.crop(x, i, j, h, w)
			y = TF.crop(y, i, j, h, w)

		
		return x, y
			