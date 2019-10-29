import neural_net

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def predict(data, net, T=50, class_count=10):
  mvn = MultivariateNormal(torch.zeros(class_count), torch.eye(class_count))
  mu, log_sigma2 = net(data)
  y_hat = torch.zeros_like(mu)
    
  for t in range(T):
    y_hat += F.softmax(mu + torch.exp(0.5*log_sigma2)*mvn.sample((len(mu),)), dim=1).detach() / T

  return y_hat / T


class Net(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, hidden_count):
    super(Net, self).__init__()
    # self.mu = neural_net.FFLayers(input_size, output_size, hidden_size, hidden_count)
    # self.log_sigma2 = neural_net.FFLayers(input_size, output_size, hidden_size, hidden_count)
    self.output_size = output_size
    self.backbone = neural_net.FFLayers(input_size, output_size*2, hidden_size, hidden_count)
  def forward(self, x):
    output = self.backbone(x)
    return output[:, :self.output_size], output[:, self.output_size:]
    # self.mu(x), self.log_sigma2(x)

class Loss(torch.nn.Module):
  def __init__(self, class_count=10, T=25):
    super(Loss,self).__init__()
    self.mvn = MultivariateNormal(torch.zeros(class_count), torch.eye(class_count))
    self.T = T
    self.class_count = class_count

  def forward(self, output, y):
    mu, log_sigma2 = output
    y_hat = []
    
    for t in range(self.T):
      epsilon = self.mvn.sample((len(mu),))
      numerator = F.log_softmax(mu + torch.exp(0.5*log_sigma2)*epsilon, dim=1)
      y_hat.append( numerator - np.log(self.T))

    y_hat = torch.stack(tuple(y_hat))
    return F.nll_loss(torch.logsumexp(y_hat, dim=0), y)
