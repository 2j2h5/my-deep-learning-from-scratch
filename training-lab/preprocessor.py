from alexnet import AlexNet
from simple_conv_net import SimpleConvNet, SimpleConvNet_CIFAR10
from lenet import LeNet5, LeNet5_CIFAR10
from vggnet import VGG16_CIFAR10
from resnet import ResNet
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class Preprocessor:
    def __init__(self, model, batch_size):
        self.train_transform, self.test_transform = self._get_transform(model)
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
        elif isinstance(model, (SimpleConvNet_CIFAR10, LeNet5_CIFAR10, VGG16_CIFAR10)):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
        elif isinstance(model, ResNet):
            train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
            ])
            return train_transform, test_transform
        else:
            raise ValueError("Unknown model")
        
    def _get_MNIST_data(self):
        data_train = datasets.MNIST('data/MNIST', train=True, download=True, transform=self.transform)
        data_test = datasets.MNIST('data/MNIST', train=False, download=True, transform=self.transform)

        data_train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        data_test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

        return data_train, data_test, data_train_loader, data_test_loader
    
    def _get_CIFAR10_data(self):
        data_train = datasets.CIFAR10('data/CIFAR10', train=True, download=True, transform=self.train_transform)
        data_test = datasets.CIFAR10('data/CIFAR10', train=False, download=True, transform=self.test_transform)

        data_train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True)
        data_test_loader = DataLoader(data_test, batch_size=self.batch_size, shuffle=False)

        return data_train, data_test, data_train_loader, data_test_loader
    
    def get_data(self, dataset):
        if dataset == 'MNIST':
            data_train, data_test, data_train_loader, data_test_laoder = self._get_MNIST_data()
        elif dataset == 'CIFAR10':
            data_train, data_test, data_train_loader, data_test_laoder = self._get_CIFAR10_data()
        else:
            raise ValueError("Unknown dataset")
        
        return data_train, data_test, data_train_loader, data_test_laoder
        