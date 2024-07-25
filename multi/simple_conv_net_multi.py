import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], conv_param['filter_num'], kernel_size=conv_param['filter_size'],
                               stride=conv_param['stride'], padding=conv_param['pad'])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        conv_output_size = (input_dim[1] - conv_param['filter_size'] + 2 * conv_param['pad']) // conv_param['stride'] + 1
        pool_output_size = conv_param['filter_num'] * (conv_output_size // 2) * (conv_output_size // 2)
        self.fc1 = nn.Linear(pool_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)