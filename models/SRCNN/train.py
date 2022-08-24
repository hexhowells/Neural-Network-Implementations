import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from tqdm import tqdm

from SRCNN import SRCNN
import hyperparameters as hp
import dataloader as loader
import utils


# set losses and clear losses log file
losses = [0]

# Load modek
model = SRCNN()
model.cuda()
utils.model_summary(model)
lossfn = nn.MSELoss()

opt = torch.optim.Adam([
			{'params': model.conv1.parameters()},
			{'params': model.conv2.parameters()},
			{'params': model.conv3.parameters(), 'lr': hp.lr * 0.1}
		], lr=hp.lr)


# load data generator
params = {'batch_size':hp.batch_size,
		  'shuffle': True}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

compose = transforms.Compose([normalize,
							  transforms.Resize(256),
							  ])

training_loader = loader.SuperResolution("E:/Image Datasets/ImageSuperResolution/SRx4", transform=compose)
training_generator = torch.utils.data.DataLoader(training_loader, **params)


for epoch in range(hp.epochs):
	avg_loss = sum(losses[-10:]) / 10
	print("Epoch: {}\tLoss: {}".format(epoch, avg_loss))
	losses = [losses[-1]]

	for x_batch, y_batch in tqdm(training_generator):
		opt.zero_grad()
		pred = model(x_batch.to(device='cuda', dtype=torch.float))

		targets = y_batch.to(device='cpu', dtype=torch.float)

		loss = lossfn(pred, targets)
		losses.append(loss.detach().item())
		loss.backward()
		opt.step()
	
	torch.save(model.state_dict(), "models/SRCNN.pth")
