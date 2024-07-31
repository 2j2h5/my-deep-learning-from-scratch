from alexnet import AlexNet
from simple_conv_net import SimpleConvNet
from lenet import LeNet5
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
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    'AlexNet': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

class Preprocessor:
    def __init__(self, model, batch_size):
        self.transform = self._get_transform(model)
        self.batch_size = batch_size

    def _get_transform(self, model):
        if isinstance(model, SimpleConvNet):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif isinstance(model, LeNet5):
            return transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif isinstance(model, AlexNet):
            return transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError(f"Unknown model")
        
    def get_MNIST_data(self):
        data_train = MNIST('.', train=True, download=True, transform=self.transform)
        data_test = MNIST('.', train=False, download=True, transform=self.transform)

        data_train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        data_test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

        return data_train, data_test, data_train_loader, data_test_loader