import sys

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ExponentialLR

import aleatoric, epistemic, combined, neural_net

train_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=True, download=True,
  transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(
datasets.MNIST('data', train=False, 
  transform=transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.1307,), (0.3081,))
])),
batch_size=256, shuffle=True)

kind = sys.argv[1]
if kind == 'epistemic':
  model = epistemic
elif kind == 'aleatoric':
  model = aleatoric
elif kind == 'combined':
  model = combined
else:
  print ('kind can be epistemic, aleatoric or combined')
  exit()


net = model.Net(28*28, 10, 1024, 2)
net.apply(neural_net.init_weights)

criterion = model.Loss()

predict = model.predict

kwargs = dict(lr=1e-4, weight_decay=0.0001) if kind == 'aleatoric' else dict(lr=1e-4)
optimizer = torch.optim.Adam(net.parameters(), **kwargs)

scheduler = ExponentialLR(optimizer, gamma=0.9999)

net.train()
for epoch in range(10):
  train_losses = neural_net.train(train_loader, net, criterion, optimizer, scheduler)  
  print ('Train loss = %s' % (sum(train_losses) / len(train_losses)) )

  score, loss = neural_net.test(test_loader, predict, net, criterion)
  print ('Testing: Accuracy = %.2f%%, Loss %.4f' % (score*100, loss))

torch.save(net, '%s.pt' % kind)