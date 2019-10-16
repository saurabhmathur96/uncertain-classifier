import torch
from os import path
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from train_epistemic import predict
from scipy.stats import entropy

net = torch.load('epistemic.pt')
test_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=False, 
  transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=256, shuffle=True)

images = []
ps = []
net.train()
for data, _ in test_loader:
	p = predict(data.view(len(data), -1), net)
	images.extend(data.numpy())
	
	ps.extend(p.view(-1, 10).detach().numpy())

uncertainties = [entropy(p) for p in ps]
plt.figure()
indices = np.argsort(uncertainties)[:49]
for i, index in enumerate(indices, start=1):
	plt.subplot(7,7,i)
	plt.imshow(images[index].reshape(28,28), cmap='gray')
	plt.axis('off')
	print (uncertainties[index])

plt.savefig('high_epistemic.png')

plt.figure()
indices = np.argsort(uncertainties)[::-1][:49]
for i, index in enumerate(indices, start=1):
	plt.subplot(7,7,i)
	plt.imshow(images[index].reshape(28,28), cmap='gray')
	plt.axis('off')
	print (uncertainties[index])

plt.savefig('low_epistemic.png')
