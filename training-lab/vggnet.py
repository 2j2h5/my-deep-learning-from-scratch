import torch.nn as nn
from collections import OrderedDict

class Convs(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(Convs, self).__init__()
        layers = []

        for i in range(num_convs):
            layers.append((f'conv{i+1}', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1)))
            layers.append((f'relu{1+1}', nn.ReLU()))
            in_channels = out_channels
        layers.append(('pool', nn.MaxPool2d(kernel_size=(2, 2), stride=2)))
        self.c = nn.Sequential(OrderedDict(layers))

    def forward(self, img):
        output = self.c(img)
        return output
    
class VGG16_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10, self).__init__()

        self.feature = nn.Sequential(
            Convs(3, 64, 2),
            Convs(64, 128, 2),
            Convs(128, 256, 3),
            Convs(256, 512, 3),
            Convs(512, 512, 3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, img):
        output = self.feature(img)
        output = output.view(img.size(0), -1)
        output = self.classifier(output)
        return output