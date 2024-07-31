from alexnet import AlexNet
from simple_conv_net import SimpleConvNet
from lenet import LeNet5
from preprocessor import Preprocessor
from trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20

net = SimpleConvNet()
preprocessor = Preprocessor(net, BATCH_SIZE)
data_train, data_test, data_train_loader, data_test_loader = preprocessor.get_MNIST_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

print(f'Use {device}')

optimizer = optim.Adam(net.parameters(), LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(net, device, data_train_loader, data_test_loader, optimizer, criterion, NUM_EPOCHS)
trainer.train()