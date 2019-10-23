import torch
from os import path
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from train_aleatoric import predict as predict_aleatoric
from scipy.stats import entropy
from tqdm import tqdm

net = torch.load('combined.pt')
test_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=False, 
  transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=1024, shuffle=False)

images = []
ps = []
classes = []
def predict(data, net, T=50):

	p = [predict_aleatoric(data, net) for _ in range(T)]
	return sum(p) / len(p)

net.train()
for data, target in tqdm(test_loader):
	p = predict(data.view(len(data), -1), net)
	images.extend(data.numpy())
	
	ps.extend(p.view(-1, 10).detach().numpy())
	classes.extend(target.view(-1).detach().numpy())

epistemic = np.array([entropy(p) for p in ps])
epistemic = (epistemic - np.min(epistemic))/np.ptp(epistemic)

aleatoric = []
net.eval()
for data, target in tqdm(test_loader):
	mu, log_sigma2 = net(data.view(len(data), -1))
	#images.extend(data.numpy())
	
	aleatoric.extend(log_sigma2.view(-1).detach().numpy())
	#classes.extend(target.view(-1).detach().numpy())
aleatoric = np.array(aleatoric)
aleatoric = (aleatoric - np.min(aleatoric))/np.ptp(aleatoric)
#uncertainties = [entropy(p) for p in ps]
classes = np.array(classes)
#all_uncertainties = np.array(uncertainties)
all_images = np.array(images)




plt.figure()
for t in range(10):
	#uncertainties = epistemic[classes == t] - aleatoric[classes == t]
	images = all_images[classes==t]
	indices = np.argsort((epistemic[classes == t] - aleatoric[classes == t])**2)[::-1]
	
	i = 0
	for index in indices:
		if epistemic[index] > aleatoric[index] >  np.percentile(aleatoric, 75) :
			i+=1
			
			plt.subplot(10,10,t*10+i)
			plt.imshow(images[index].reshape(28,28), cmap='gray')
			plt.axis('off')
			#print (uncertainties[index])
			
			if i == 10:
				break

plt.savefig('low_combined_aleatoric.png')





plt.figure()
for t in range(10):
	#uncertainties = epistemic[classes == t] - aleatoric[classes == t]
	images = all_images[classes==t]
	indices = np.argsort((epistemic[classes == t] - aleatoric[classes == t])**2)[::-1]
	
	i = 0
	for index in indices:
		if  aleatoric[index] > epistemic[index] > np.percentile(epistemic, 75):
			i+=1
			
			plt.subplot(10,10,t*10+i)
			plt.imshow(images[index].reshape(28,28), cmap='gray')
			plt.axis('off')
			#print (uncertainties[index])
			
			if i == 10:
				break

plt.savefig('low_combined_epistemic.png')


plt.figure()
for t in range(10):
	#uncertainties = epistemic[classes == t] - aleatoric[classes == t]
	images = all_images[classes==t]
	indices = np.argsort((epistemic[classes == t] - aleatoric[classes == t])**2)
	
	i = 0
	for index in indices:
		if epistemic[index] < np.percentile(epistemic, 25):
			i+=1
			
			plt.subplot(10,10,t*10+i)
			plt.imshow(images[index].reshape(28,28), cmap='gray')
			plt.axis('off')
			#print (uncertainties[index])
			
			if i == 10:
				break

plt.savefig('high_combined.png')