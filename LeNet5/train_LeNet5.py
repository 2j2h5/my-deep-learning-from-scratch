from LeNet5 import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_train = MNIST('.',
                   train=True,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('.',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=1024, shuffle=False)

net = LeNet5()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

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

for epoch in range(1, 21):
    train(net, device, data_train_loader, optimizer, epoch)
    test(net, device, data_test_loader)