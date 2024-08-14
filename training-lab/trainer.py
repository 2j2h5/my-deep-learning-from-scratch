import torch
import time

class Trainer:
    def __init__(self, network, device, data_train_loader, data_test_loader, optimizer, lr_scheduler, criterion, epoch):
        self.network = network
        self.device = device
        self.data_train_loader = data_train_loader
        self.data_test_loader = data_test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.epoch = epoch

        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self, step):
        self.network.train()
        total_correct = 0
        total_loss = 0.0

        for batch_index, (images, labels) in enumerate(self.data_train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            output = self.network(images)

            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(labels.view_as(pred)).sum().item()

            if batch_index % 100 == 0:
                print(f'Train Epoch: {step}, [{batch_index * len(images)}/{len(self.data_train_loader.dataset)} ({100. * batch_index / len(self.data_train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.data_train_loader.dataset)
        accuracy = total_correct / len(self.data_train_loader.dataset)
        self.train_acc_list.append(accuracy)
        print(f'\nTrain set: Average loss: {avg_loss:.4f}, Accuracy: {total_correct}/{len(self.data_train_loader.dataset)} ({100. * accuracy:.0f}%)')

    def test_step(self):
        self.network.eval()
        total_correct = 0
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in self.data_test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.network(images)
                test_loss += self.criterion(output, labels).item()
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(labels.view_as(pred)).sum().item()

        test_loss /= len(self.data_test_loader.dataset)
        accuracy = total_correct / len(self.data_test_loader.dataset)
        self.test_acc_list.append(accuracy)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {total_correct}/{len(self.data_test_loader.dataset)} ({100. * accuracy:.0f}%)')

        return test_loss

    def train(self):
        start_time = time.time()
        for step in range(1, self.epoch+1):
            self.train_step(step)
            self.test_step()
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            print(f"Learning Rate: {current_lr:.6f}\n")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total Training Time: {elapsed_time:.2f} seconds")

    def get_acc_list(self):
        return self.train_acc_list, self.test_acc_list