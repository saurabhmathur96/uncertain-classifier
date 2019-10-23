import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def init_weights(m):
  if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)

def FFLayer(input_size, output_size):
  return nn.Sequential(*[
    nn.Linear(input_size, output_size),
    nn.ReLU()
  ])

def FFLayers(input_size, output_size, hidden_size, hidden_count):
  return nn.Sequential(*[
    FFLayer(input_size, hidden_size),
    *[FFLayer(hidden_size, hidden_size) for i in range(hidden_count)],
    nn.Linear(hidden_size, output_size)
  ])


class FFDropoutLayer(torch.nn.Module):

  def __init__(self, input_size, output_size, dropout_p=.2):
    super(FFDropoutLayer,self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    self.dropout_p = dropout_p
  
  def forward(self, x):
    x = self.linear(x)
    x = F.dropout(x, p=self.dropout_p, training=self.training)
    return F.relu(x)
    

def FFDropoutLayers(input_size, output_size, hidden_size, hidden_count, dropout_p=.2):
  return nn.Sequential(*[
    FFDropoutLayer(input_size, hidden_size),
    *[FFDropoutLayer(hidden_size, hidden_size) for i in range(hidden_count)],
    nn.Linear(hidden_size, output_size)
  ])


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

def test(test_loader, predict, net, criterion):
  scores = []
  losses = []
  net.train()
  for data, target in test_loader:
    output = predict(data.view(len(data), -1), net)
    
    predictions = torch.argmax(output.data, 1)
    scores.append((predictions == target).float().mean().detach().item())
    
    loss = criterion(net(data.view(len(data), -1)), target)
    losses.append(loss.detach().item())
  return sum(scores)/len(test_loader), sum(losses)/len(test_loader)