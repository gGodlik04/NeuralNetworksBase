import pandas as pd

from sklearn import model_selection, datasets, metrics
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

%matplotlib inline
%pylab inline

data = datasets.load_breast_cancer()

print(data.keys())
print(data.target.shape)

data.frame

#Описание датасета
print(data.DESCR)


df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.head()

#проверить данные на пропуски и корректность
df.info()

#Подготовка датасета
X = data.data
y = data.target
# разделим данные с помощью Scikit-Learn's train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)

from sklearn.metrics import accuracy_score

def measure_quality(predictions):
    return accuracy_score(y_test, predictions)

model = nn.Sequential(
    nn.Linear(30, 64),
    nn.ReLU(),
    nn.Linear(64,2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение нейронной сети
loss_values = []
accuracy_values = []

num_epochs = 40
for epoch in range(num_epochs):
    inputs = torch.tensor(X_train, dtype=torch.float32)
    labels = torch.tensor(y_train, dtype=torch.long)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())

    with torch.no_grad():
        inputs_test = torch.tensor(X_test, dtype=torch.float32)
        labels_test = torch.tensor(y_test, dtype=torch.long)
        outputs_test = model(inputs_test)
        _, predicted = torch.max(outputs_test, 1)
        accuracy = measure_quality(predicted)
        accuracy_values.append(accuracy.item())
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Графики Loss и Accuracy
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(accuracy_values, label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Инференс
with torch.no_grad():
    inputs = torch.tensor(X_test[1], dtype=torch.float32)
    output = model(inputs)
    predicted = torch.argmax(output).item()

print("Predicted class:", predicted)
print("True class:", y_test[1])