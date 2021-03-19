import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from adversarialbox.attacks import LinfPGDAttack
from adversarialbox.train import adv_train
from adversarialbox.utils import to_var, pred_batch, test

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(

            torch.nn.Conv2d(1, 16, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc = torch.nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

param = {
    'batch_size': 100,
    'test_batch_size': 100,
    'num_epochs': 15,
    'delay': 10,
    'learning_rate': 1e-3,
    'weight_decay': 5e-4,
}

pretrained_model='./cnn.pkl'

train_dataset = datasets.MNIST(root='../data/',train=True, download=True,
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset,
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True,
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset,
    batch_size=param['test_batch_size'], shuffle=True)

net=CNN()

net.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

net.eval()

if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()

adversary = LinfPGDAttack()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'],
    weight_decay=param['weight_decay'])

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(loader_train):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, loader_test)

torch.save(net.state_dict(), 'adv_trained_cnn.pkl')

