import neural_net

import torch
from torch import nn
import torch.nn.functional as F

def predict(data, net, T=25, class_count=10):
  predictions = []
  for t in range(T):
    predictions.append(F.softmax(net(data), dim=1))
  return sum(predictions)/len(predictions)


class Net(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, hidden_count):
    super(Net, self).__init__()
    self.backbone = neural_net.FFDropoutLayers(input_size, output_size, hidden_size, hidden_count)
    
  def forward(self, x):
    return self.backbone(x)

Loss = nn.CrossEntropyLoss
