import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import numpy as np

def init_weights(m):
  if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.01)
      
def FFLayer(input_size, output_size):
  return nn.Sequential(*[
    nn.Linear(input_size, output_size),
    nn.ReLU()
  ])

def FFDropoutLayer(input_size, output_size, dropout_p=.2):
  return nn.Sequential(*[
    nn.Linear(input_size, output_size),
    nn.Dropout(dropout_p),
    nn.ReLU()
  ])


class AleatoricNN(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, hidden_count):
    super(AleatoricNN, self).__init__()
    self.mu = nn.Sequential(*[
      FFLayer(input_size, hidden_size),
      *[FFLayer(hidden_size, hidden_size) for i in range(hidden_count)],
      nn.Linear(hidden_size, output_size)
    ])
    self.log_sigma2 = nn.Sequential(*[
      FFLayer(input_size, hidden_size),
      *[FFLayer(hidden_size, hidden_size) for i in range(hidden_count)],
      nn.Linear(hidden_size, 1)
    ])
  def forward(self, x):
    return self.mu(x), self.log_sigma2(x)
    
class AleatoricLoss(torch.nn.Module):
  def __init__(self, class_count=10, T=25):
    super(AleatoricLoss,self).__init__()
    self.mvn = MultivariateNormal(torch.zeros(class_count), torch.eye(class_count))
    self.T = T
    self.class_count = class_count

  def forward(self, mu, log_sigma2, y):
    y_hat = []
    
    for t in range(self.T):
      y_hat.append( F.log_softmax(mu + torch.exp(log_sigma2)*self.mvn.sample((len(mu),)), dim=1) - np.log(self.T))

    return F.nll_loss(torch.logsumexp(torch.stack(tuple(y_hat)), dim=0), y)

def EpistemicNN(input_size, output_size, hidden_size, hidden_count):
  return nn.Sequential(*[
      FFDropoutLayer(input_size, hidden_size),
      *[FFDropoutLayer(hidden_size, hidden_size) for i in range(hidden_count)],
      nn.Linear(hidden_size, output_size)
    ])


class CombinedNN(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, hidden_count):
    super(CombinedNN, self).__init__()
    self.backbone = nn.Sequential(*[
      FFDropoutLayer(input_size, hidden_size),
      *[FFDropoutLayer(hidden_size, hidden_size) for i in range(hidden_count)]
    ])
    self.mu = nn.Linear(hidden_size, output_size)
    self.log_sigma2 = nn.Linear(hidden_size, 1)

  def forward(self, x):
    embedded = self.backbone(x)
    return self.mu(embedded), self.log_sigma2(embedded)

def accuracy_score(p, y):
  _, predicted = torch.max(p.data, 1)
  total = y.size(0)
  correct = (predicted == y).sum().item()
  return correct/total