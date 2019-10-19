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
classes =[]
net.eval()
for data, target in test_loader:
	mu, log_sigma2 = net(data.view(len(data), -1))
	images.extend(data.numpy())
	
	uncertainties.extend(log_sigma2.view(-1).detach().numpy())
	classes.extend(target.view(-1).detach().numpy())

classes = np.array(classes)
all_uncertainties = np.array(uncertainties)
all_images = np.array(images)
plt.figure()
for t in range(10):
	uncertainties = all_uncertainties[classes == t]
	images = all_images[classes==t]
	indices = np.argsort(uncertainties)[:10]
	for i, index in enumerate(indices):
		plt.subplot(10,10,t*10+i+1)
		plt.imshow(images[index].reshape(28,28), cmap='gray')
		plt.axis('off')
		print (uncertainties[index])

plt.savefig('high_aleatoric.png')

plt.figure()
for t in range(10):
	uncertainties = all_uncertainties[classes == t]
	images = all_images[classes==t]
	indices = np.argsort(uncertainties)[::-1][:10]
	for i, index in enumerate(indices):
		plt.subplot(10,10,t*10+i+1)
		plt.imshow(images[index].reshape(28,28), cmap='gray')
		plt.axis('off')
		print (uncertainties[index])

plt.savefig('low_aleatoric.png')