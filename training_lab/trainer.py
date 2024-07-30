from alexnet import AlexNet
from simple_conv_net import SimpleConvNet
from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = {
    'SimpleConvNet': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'LeNet5': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]),
    'AlexNet': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

data_train = MNIST('.', train=True, download=True, transform=transform['AlexNet'])
data_test = MNIST('.', train=False, download=True, transform=transform['AlexNet'])

data_train_loader = DataLoader(data_train, batch_size=100, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=100, shuffle=False)

net = AlexNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Use {device}')
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(net, device, data_train_loader, optimizer, epoch):
    net.train()
    for batch_index, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            print(f'Train Epoch: {epoch}, [{batch_index * len(images)}/{len(data_train_loader.dataset)} ({100. * batch_index / len(data_train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(net, device, data_test_loader):
    net.eval()
    total_correct = 0
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in data_test_loader:
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            test_loss += criterion(output, labels).item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(data_test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {total_correct}/{len(data_test_loader.dataset)} ({100. * total_correct / len(data_test_loader.dataset):.0f}%)\n')

torch.autograd.set_detect_anomaly(True)
for epoch in range(1, 21):
    train(net, device, data_train_loader, optimizer, epoch)
    test(net, device, data_test_loader)