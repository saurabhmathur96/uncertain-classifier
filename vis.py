import aleatoric, epistemic, combined, neural_net

import torch
from torchvision import transforms, datasets
from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys

def show_digits(images, w, h):
  for i, image in enumerate(images):
    plt.subplot(w, h, i+1)
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.axis('off')

def slice_by_digit(images, digits, uncertainty, n=10):
  high = []
  low = []
  for i in range(1, 10+1):
    images_i = images[digits == i]
    uncertainty_i = uncertainty[digits == i]
    indices = uncertainty_i.argsort()
    high.extend(images_i[indices[::-1][:n]])
    low.extend(images_i[indices[:n]])
  return high, low

if __name__ == '__main__':
  variant = sys.argv[1]

  test_loader = torch.utils.data.DataLoader(
  datasets.MNIST('data', train=False, 
    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])),
  batch_size=256, shuffle=False)


  if variant == 'combined':
    net = torch.load('combined.pt')
    epistemic, aleatoric = [], []
    images, digits = [], []
    with torch.no_grad():
      for data, target in tqdm(test_loader):
        x = data.view(len(data), -1).float()
        mu, log_sigma2 = net(x)
        aleatoric.extend([ np.linalg.norm(s) for s in np.exp(0.5*log_sigma2.detach().numpy()) ])
        epistemic.extend([ entropy(p) for p in combined.predict(x, net).detach().numpy() ])
        images.extend(data.detach().numpy())
        digits.extend(target.detach().numpy())

    images, digits = np.array(images), np.array(digits)
    epistemic, aleatoric = np.array(epistemic), np.array(aleatoric)


    high, low = slice_by_digit(images, digits, epistemic)

    plt.figure()
    show_digits(high, 10, 10)
    plt.savefig('combined_epistemic_high.png')

    plt.figure()
    show_digits(low, 10, 10)
    plt.savefig('combined_epistemic_low.png')


    high, low = slice_by_digit(images, digits, aleatoric)

    plt.figure()
    show_digits(high, 10, 10)
    plt.savefig('combined_aleatoric_high.png')

    plt.figure()
    show_digits(low, 10, 10)
    plt.savefig('combined_aleatoric_low.png')

  elif variant == 'epistemic':
    net = torch.load('epistemic.pt')
    predict = epistemic.predict
    epistemic = []
    images, digits = [], []

    with torch.no_grad():
      for data, target in tqdm(test_loader):
        x = data.view(len(data), -1).float()
        epistemic.extend([ entropy(p) for p in predict(x, net).detach().numpy() ])
        images.extend(data.detach().numpy())
        digits.extend(target.detach().numpy())

      images, digits = np.array(images), np.array(digits)
      epistemic= np.array(epistemic)

    high, low = slice_by_digit(images, digits, epistemic)

    plt.figure()
    show_digits(high, 10, 10)
    plt.savefig('epistemic_high.png')

    plt.figure()
    show_digits(low, 10, 10)
    plt.savefig('epistemic_low.png')

  elif variant == 'aleatoric':
    net = torch.load('aleatoric.pt')
    aleatoric = []
    images, digits = [], []

    with torch.no_grad():
      for data, target in tqdm(test_loader):
        x = data.view(len(data), -1).float()
        mu, log_sigma2 = net(x)
        aleatoric.extend([ np.linalg.norm(s) for s in np.exp(0.5*log_sigma2.detach().numpy()) ])
        images.extend(data.detach().numpy())
        digits.extend(target.detach().numpy())

      images, digits = np.array(images), np.array(digits)
      aleatoric = np.array(aleatoric)

    high, low = slice_by_digit(images, digits, aleatoric)

    plt.figure()
    show_digits(high, 10, 10)
    plt.savefig('aleatoric_high.png')

    plt.figure()
    show_digits(low, 10, 10)
    plt.savefig('aleatoric_low.png')

  else:
    print ('variant must be one of combined,aleatoric,epistemic')