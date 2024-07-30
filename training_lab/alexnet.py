import torch.nn as nn
from collections import OrderedDict

class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)),
            ('relu1', nn.ReLU()),
            ('n1', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),
            ('s1', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output
    
class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(96, 256, 5, padding=2)),
            ('relu2', nn.ReLU()),
            ('n2', nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)),
            ('s2', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output
    
class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(256, 384, 3, padding=1)),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output
    
class C4(nn.Module):
    def __init__(self):
        super(C4, self).__init__()

        self.c4 = nn.Sequential(OrderedDict([
            ('c4', nn.Conv2d(384, 384, 3, padding=1)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c4(img)
        return output
    
class C5(nn.Module):
    def __init__(self):
        super(C5, self).__init__()

        self.c5 = nn.Sequential(OrderedDict([
            ('c5', nn.Conv2d(384, 256, 3, padding=1)),
            ('relu5', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=3, stride=2))
        ]))

    def forward(self, img):
        output = self.c5(img)
        return output
    
class F6(nn.Module):
    def __init__(self):
        super(F6, self).__init__()

        self.f6 = nn.Sequential(OrderedDict([
            ('drop6', nn.Dropout(p=0.5)),
            ('f6', nn.Linear(in_features=(256 * 6 * 6), out_features=4096)),
            ('relu6', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f6(img)
        return output
    
class F7(nn.Module):
    def __init__(self):
        super(F7, self).__init__()

        self.f7 = nn.Sequential(OrderedDict([
            ('drop7', nn.Dropout(p=0.5)),
            ('f7', nn.Linear(in_features=4096, out_features=4096)),
            ('relu7', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f7(img)
        return output
    
class F8(nn.Module):
    def __init__(self):
        super(F8, self).__init__()

        self.f8 = nn.Sequential(OrderedDict([
            ('f8', nn.Linear(in_features=4096, out_features=10)),
            ('sm8', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.f8(img)
        return output

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.c1 = C1()
        self.c2 = C2()
        self.c3 = C3()
        self.c4 = C4()
        self.c5 = C5()
        self.f6 = F6()
        self.f7 = F7()
        self.f8 = F8()

    def init_bias(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.constant_(self.c2.c2[0].bias, 1)
        nn.init.constant_(self.c4.c4[0].bias, 1)
        nn.init.constant_(self.c5.c5[0].bias, 1)

    def forward(self, img):
        output = self.c1(img)
        output = self.c2(output)
        output = self.c3(output)
        output = self.c4(output)
        output = self.c5(output)
        output = output.view(-1, 256*6*6)
        output = self.f6(output)
        output = self.f7(output)
        output = self.f8(output)
        return output