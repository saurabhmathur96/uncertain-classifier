import torch
from torch import nn


import neural_net, aleatoric




class Net(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, hidden_count):
    super(Net, self).__init__()
    self.mu = neural_net.FFDropoutLayers(input_size, output_size, hidden_size, hidden_count)
    self.log_sigma2 = neural_net.FFDropoutLayers(input_size, 1, hidden_size, hidden_count)
  
  def forward(self, x):
    return self.mu(x), self.log_sigma2(x)

def predict(data, net, T=25, class_count=10):
  predictions = []
  for t in range(T):
    predictions.append(aleatoric.predict(data, net))
  return sum(predictions)/len(predictions)

Loss = aleatoric.Loss