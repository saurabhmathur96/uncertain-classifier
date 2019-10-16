import torch
from torch import nn
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import ExponentialLR
from tqdm.auto import tqdm
from models import *


def train(train_loader, net, criterion, optimizer, scheduler):
  train_losses = []
  for data, target in tqdm(train_loader):
    data, target = data.view(len(data), -1).float(), target
    optimizer.zero_grad()
    output = net(data)
    loss = criterion(output, target)
    train_losses.append(loss.item())
    loss.backward()
    optimizer.step()
  
  return train_losses

def predict(data, net, T=25, class_count=10):
  predictions = []
  for t in range(T):
    predictions.append(F.softmax(net(data), dim=1))
  return sum(predictions)/len(predictions)

def test(test_loader, net, criterion):
  scores = []
  losses = []
  for data, target in test_loader:
    net.train()
    output = predict(data.view(len(data), -1), net)
    
    predictions = torch.argmax(output.data, axis=1)
    scores.append((predictions == target).float().mean().detach().item())
    
    loss = criterion(net(data.view(len(data), -1)), target)
    losses.append(loss.detach().item())
  return sum(scores)/len(test_loader), sum(losses)/len(test_loader)


if __name__ == '__main__':
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

  # input_size, output_size, hidden_size, hidden_count
  net = EpistemicNN(28*28, 10, 1024, 5)
  net.apply(init_weights)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.001)
  scheduler = ExponentialLR(optimizer, gamma=0.9999)


  for epoch in range(10):
    net.train()
    train_losses = train(train_loader, net, criterion, optimizer, scheduler)  
    print ('Train loss = %s' % (sum(train_losses) / len(train_losses)) )

    score, loss = test(test_loader, net, criterion)
    print ('Testing: Accuracy = %.2f%%, Loss %.4f' % (score*100, loss))

  torch.save(net, 'epistemic.pt')