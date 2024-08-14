from alexnet import AlexNet 
from simple_conv_net import SimpleConvNet, SimpleConvNet_CIFAR10
from lenet import LeNet5, LeNet5_CIFAR10
from vggnet import VGG16_CIFAR10
from resnet import *

from preprocessor import Preprocessor
from trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
BATCH_SIZE = 256
NUM_EPOCHS = 30
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

net = resnet20()
preprocessor = Preprocessor(net, BATCH_SIZE)
data_train, data_test, data_train_loader, data_test_loader = preprocessor.get_data('CIFAR10')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

print(f'Using {device}')

optimizer = optim.SGD(net.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, threshold=1e-4)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(net, device, data_train_loader, data_test_loader, optimizer, lr_scheduler, criterion, NUM_EPOCHS)
trainer.train()
train_acc_list, test_acc_list = trainer.get_acc_list()

x = np.arange(NUM_EPOCHS)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()