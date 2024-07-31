import torch.nn as nn
from collections import OrderedDict

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=0)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output
        
class F2(nn.Module):
    def __init__(self):
        super(F2, self).__init__()

        self.f2 = nn.Sequential(OrderedDict([
            ('f2', nn.Linear(4320, 100)),
            ('relu2', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f2(img)
        return output

class F3(nn.Module):
    def __init__(self):
        super(F3, self).__init__()

        self.f3 = nn.Sequential(OrderedDict([
            ('f3', nn.Linear(100, 10)),
            ('sm3', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f3(img)
        return output

class SimpleConvNet(nn.Module):
    def __init__(self):
        super(SimpleConvNet, self).__init__()

        self.c1 = C1()
        self.f2 = F2()
        self.f3 = F3()

    def forward(self, img):
        output = self.c1(img)
        output = output.view(img.size(0), -1)
        output = self.f2(output)
        output = self.f3(output)
        return output
    
class C1_CIFAR10(nn.Module):
    def __init__(self):
        super(C1_CIFAR10, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=0)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output
        
class F2_CIFAR10(nn.Module):
    def __init__(self):
        super(F2_CIFAR10, self).__init__()

        self.f2 = nn.Sequential(OrderedDict([
            ('f2', nn.Linear(30*14*14, 100)),
            ('relu2', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f2(img)
        return output

class F3_CIFAR10(nn.Module):
    def __init__(self):
        super(F3_CIFAR10, self).__init__()

        self.f3 = nn.Sequential(OrderedDict([
            ('f3', nn.Linear(100, 10)),
            ('sm3', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f3(img)
        return output

class SimpleConvNet_CIFAR10(nn.Module):
    def __init__(self):
        super(SimpleConvNet_CIFAR10, self).__init__()

        self.c1 = C1_CIFAR10()
        self.f2 = F2_CIFAR10()
        self.f3 = F3_CIFAR10()

    def forward(self, img):
        output = self.c1(img)
        output = output.view(img.size(0), -1)
        output = self.f2(output)
        output = self.f3(output)
        return output