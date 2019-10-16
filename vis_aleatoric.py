import torch
from os import path
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

net = torch.load('aleatoric.pt')
test_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=False, 
  transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=256, shuffle=True)

images = []
uncertainties = []
net.eval()
for data, _ in test_loader:
	mu, log_sigma2 = net(data.view(len(data), -1))
	images.extend(data.numpy())
	
	uncertainties.extend(log_sigma2.view(-1).detach().numpy())

plt.figure()
indices = np.argsort(uncertainties)[:49]
for i, index in enumerate(indices, start=1):
	plt.subplot(7,7,i)
	plt.imshow(images[index].reshape(28,28), cmap='gray')
	plt.axis('off')
	print (uncertainties[index])

plt.savefig('high_aleatoric.png')

plt.figure()
indices = np.argsort(uncertainties)[::-1][:49]
for i, index in enumerate(indices, start=1):
	plt.subplot(7,7,i)
	plt.imshow(images[index].reshape(28,28), cmap='gray')
	plt.axis('off')
	print (uncertainties[index])

plt.savefig('low_aleatoric.png')