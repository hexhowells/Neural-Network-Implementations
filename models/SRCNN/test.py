import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from SRCNN import SRCNN
import dataloader as loader
import utils


model = SRCNN()
model.cuda()
model.load_state_dict(torch.load("models/SRCNN.pth"))
model.eval() # turn off dropout


params = {'batch_size':1, 'shuffle': True}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([normalize,
							  transforms.Resize(256),
							  ])

filepath = "E:/Image Datasets/ImageSuperResolution/SRx4"

training_loader = loader.SuperResolution(filepath, transform=compose, crop=False)
training_generator = torch.utils.data.DataLoader(training_loader, **params)



for x, y in training_generator:
	model_output = model(x.to(device='cuda', dtype=torch.float))
	
	x_img = utils.renderize(x)
	y_img = utils.renderize(y)
	out_img = utils.renderize(model_output)

	fig, axarr = plt.subplots(1,3)
	axarr[0].imshow(x_img)
	axarr[1].imshow(out_img)
	axarr[2].imshow(y_img)
	
	plt.show()

