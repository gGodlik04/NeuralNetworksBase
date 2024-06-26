# -*- coding: utf-8 -*-
"""Mnist(Conv classification numbers).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12fyV35cHDxeCrj_SJal6XUsueC-jQiuu
"""

import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),  # Преобразование изображений в тензоры
    transforms.Normalize((0.5,), (0.5,))  # Нормализация изображений
])

data_train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
data_test = datasets.MNIST(root="./data/", transform=transform, train = False)

from torch.utils.data import DataLoader

train_data_loader = DataLoader(data_train, batch_size = 64, shuffle=False)
test_data_loader = DataLoader(data_test, batch_size = 64, shuffle=False)

from torch import nn

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    self.classifier = nn.Sequential(
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(-1, 64*7*7)
    x = self.classifier(x)
    return x

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epoches = 1

for epoch in range(epoches):
  running_loss = 0.0

  for i, data in enumerate(train_data_loader, 0):
    inputs, labels = data

    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss +=loss.item()

    if i % 2000 == 1999:
      print('[%d, %5d] loss: $.3f' %
            (epoch + 1, i+1, running_loss / 2000))
      running_loss = 0.0

model.eval()

correct = 0
total = 0

with torch.no_grad():
  for images, labels in test_data_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('accuracy = %d %%' %(100 * correct / total))

import matplotlib.pyplot as plt
from torch.utils.data import Subset

subset_test = Subset(data_test, range(15))

for image, label in subset_test:
  outputs = model(image.unsqueeze(0))  # Добавляем размерность батча
  _, predicted = torch.max(outputs, 1)

  # Отображение изображения и метки
  image = image.squeeze()
  plt.imshow(image, cmap='gray')
  plt.title('Predicted value: %d, True value: %d' % (predicted.item(), label))

  plt.show()

