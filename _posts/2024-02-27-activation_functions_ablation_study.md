---
layout: post
title: "Learning-By-Doing Series: Understanding activation functions"
author: "Amar P"
categories: journal
tags: [neural-networks,fundamentals]
---

## Ablation study: Activation functions

**Goal here is to observ or record changes in accuracy, loss, or other performance indicators.**
- Activation Functions under consideration
  - **Sigmoid**
  - **Tanh**
  - **ReLU**
  - **LeakyReLU**
  - **ELU**
  - **SELU**
  - **Softplus**
  - **Mish**
  - **GELU**
  - **Swish**
- We are using **ResNet-18 model** with **CIFAR-10 dataset**
- Defined ResNet model accepts activatio function as argument in constructor
- We are keeping following config constant for experiement.
  - **Loss function : CrossEntropyLoss()**
  - **Optimizer : Adam**
  - **Learning Rate : 0.01**
  


```python
# import required libs
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import time
import matplotlib.pyplot as plt
```

### Load CIFAR-10 dataset


```python

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
```


### Define ResNet-18 Model

```python

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, activation=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.activation(x)
        return x
    

class ResNet_18(nn.Module):

    def __init__(self, image_channels=3, num_classes=10, activation=None):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, activation=self.activation),
            Block(out_channels, out_channels, activation=self.activation)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
```

### Define Train and test function to record loss and accuracy over epochs


```python
import time

# 3. Training and Testing functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return total_loss / len(train_loader), accuracy


def test(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(test_loader), accuracy

```


```python

# 4. Plot Loss and Accuracy
def plot_loss_and_accuracy(train_loss, test_loss, train_accuracy, test_accuracy):
    plt.figure(figsize=(20,20))
    plt.subplot(2, 2, 1)
    for key, value in train_loss.items():
        plt.plot(value, label=key)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for key, value in test_loss.items():
        plt.plot(value, label=key)
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    for key, value in train_accuracy.items():
        plt.plot(value, label=key)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    for key, value in test_accuracy.items():
        plt.plot(value, label=key)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # plt.tight_layout()
    plt.show()

```

### Integrated train-test-plot for each activation function


```python
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
```


```python

def train_and_test(activation_functions, train_loader, test_loader, criterion, device, num_epochs=NUM_EPOCHS):
    train_loss_history = {name: [] for name in activation_functions.keys()}
    test_loss_history = {name: [] for name in activation_functions.keys()}
    train_accuracy_history = {name: [] for name in activation_functions.keys()}
    test_accuracy_history = {name: [] for name in activation_functions.keys()}
        
    for activation_name, activation in activation_functions.items():
        print(f"Running with {activation_name} activation")
        
        model = ResNet_18(activation=activation).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = test(model, test_loader, criterion, device)

            print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%')

            train_loss_history[activation_name].append(train_loss)
            test_loss_history[activation_name].append(test_loss)
            train_accuracy_history[activation_name].append(train_accuracy)
            test_accuracy_history[activation_name].append(test_accuracy)

        end_time = time.time()
        total_time = end_time - start_time
        print(f'Total time taken for training and testing: {total_time:.2f} seconds')

    plot_loss_and_accuracy(train_loss_history, test_loss_history, train_accuracy_history, test_accuracy_history)

```


```python

class Swish(nn.Module):
    """Swish activation function

    Args:
        beta (float): beta parameter of the activation function
    """
    def __init__(self, beta=1.0):
        """
        Init method.
        """
        super(Swish, self).__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * self.sigmoid(self.beta * input)

activation_functions = {
    'Sigmoid': nn.Sigmoid(),
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(negative_slope=0.02),
    'ELU': nn.ELU(),
    'SELU': nn.SELU(),
    'Softplus': nn.Softplus(),
    'Mish': nn.Mish(),
    'GELU': nn.GELU(),
    'Swish': Swish()
}

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Instantiate the model with different activation functions
train_and_test(activation_functions, train_loader, test_loader, criterion, device)

```

    Running with Sigmoid activation
    Epoch 1/10: Train Loss: 1.7626, Train Accuracy: 34.99%, Test Loss: 2.6261, Test Accuracy: 16.52%
    Epoch 2/10: Train Loss: 1.3528, Train Accuracy: 50.40%, Test Loss: 2.8638, Test Accuracy: 35.46%
    Epoch 3/10: Train Loss: 1.1619, Train Accuracy: 58.13%, Test Loss: 2.4072, Test Accuracy: 27.71%
    Epoch 4/10: Train Loss: 1.0359, Train Accuracy: 63.06%, Test Loss: 3.3053, Test Accuracy: 30.67%
    Epoch 5/10: Train Loss: 0.9533, Train Accuracy: 66.18%, Test Loss: 1.1353, Test Accuracy: 60.65%
    Epoch 6/10: Train Loss: 0.8886, Train Accuracy: 68.72%, Test Loss: 3.8299, Test Accuracy: 42.87%
    Epoch 7/10: Train Loss: 0.8320, Train Accuracy: 70.79%, Test Loss: 3.1457, Test Accuracy: 30.49%
    Epoch 8/10: Train Loss: 0.7806, Train Accuracy: 72.56%, Test Loss: 3.3456, Test Accuracy: 26.51%
    Epoch 9/10: Train Loss: 0.7352, Train Accuracy: 74.09%, Test Loss: 2.7374, Test Accuracy: 36.82%
    Epoch 10/10: Train Loss: 0.6933, Train Accuracy: 75.80%, Test Loss: 2.2359, Test Accuracy: 39.73%
    Total time taken for training and testing: 154.31 seconds
    Running with Tanh activation
    Epoch 1/10: Train Loss: 1.8240, Train Accuracy: 33.67%, Test Loss: 1.5043, Test Accuracy: 45.20%
    Epoch 2/10: Train Loss: 1.3771, Train Accuracy: 50.35%, Test Loss: 1.2400, Test Accuracy: 54.29%
    Epoch 3/10: Train Loss: 1.1568, Train Accuracy: 58.75%, Test Loss: 1.0773, Test Accuracy: 61.71%
    Epoch 4/10: Train Loss: 1.0035, Train Accuracy: 64.56%, Test Loss: 1.0530, Test Accuracy: 63.07%
    Epoch 5/10: Train Loss: 0.8909, Train Accuracy: 68.73%, Test Loss: 0.9397, Test Accuracy: 67.07%
    Epoch 6/10: Train Loss: 0.8053, Train Accuracy: 71.69%, Test Loss: 0.8754, Test Accuracy: 69.44%
    Epoch 7/10: Train Loss: 0.7312, Train Accuracy: 74.40%, Test Loss: 0.8848, Test Accuracy: 69.73%
    Epoch 8/10: Train Loss: 0.6683, Train Accuracy: 76.73%, Test Loss: 0.7996, Test Accuracy: 73.06%
    Epoch 9/10: Train Loss: 0.6000, Train Accuracy: 78.94%, Test Loss: 0.8625, Test Accuracy: 71.83%
    Epoch 10/10: Train Loss: 0.5454, Train Accuracy: 80.98%, Test Loss: 0.8071, Test Accuracy: 73.30%
    Total time taken for training and testing: 153.26 seconds
    Running with ReLU activation
    Epoch 1/10: Train Loss: 1.7158, Train Accuracy: 39.29%, Test Loss: 1.3906, Test Accuracy: 49.69%
    Epoch 2/10: Train Loss: 1.1611, Train Accuracy: 58.22%, Test Loss: 1.1081, Test Accuracy: 61.40%
    Epoch 3/10: Train Loss: 0.9100, Train Accuracy: 68.14%, Test Loss: 0.9016, Test Accuracy: 68.86%
    Epoch 4/10: Train Loss: 0.7712, Train Accuracy: 73.01%, Test Loss: 0.8109, Test Accuracy: 71.62%
    Epoch 5/10: Train Loss: 0.6659, Train Accuracy: 76.60%, Test Loss: 0.7933, Test Accuracy: 72.74%
    Epoch 6/10: Train Loss: 0.5758, Train Accuracy: 80.11%, Test Loss: 0.8060, Test Accuracy: 72.81%
    Epoch 7/10: Train Loss: 0.4875, Train Accuracy: 83.03%, Test Loss: 0.7568, Test Accuracy: 76.06%
    Epoch 8/10: Train Loss: 0.4092, Train Accuracy: 85.63%, Test Loss: 0.7876, Test Accuracy: 75.12%
    Epoch 9/10: Train Loss: 0.3337, Train Accuracy: 88.33%, Test Loss: 0.8362, Test Accuracy: 74.68%
    Epoch 10/10: Train Loss: 0.2761, Train Accuracy: 90.35%, Test Loss: 0.8497, Test Accuracy: 75.54%
    Total time taken for training and testing: 152.85 seconds
    Running with LeakyReLU activation
    Epoch 1/10: Train Loss: 1.7220, Train Accuracy: 38.67%, Test Loss: 1.4768, Test Accuracy: 45.44%
    Epoch 2/10: Train Loss: 1.1572, Train Accuracy: 58.65%, Test Loss: 2.6726, Test Accuracy: 53.20%
    Epoch 3/10: Train Loss: 0.9750, Train Accuracy: 65.82%, Test Loss: 0.9273, Test Accuracy: 68.11%
    Epoch 4/10: Train Loss: 0.8102, Train Accuracy: 71.71%, Test Loss: 0.8316, Test Accuracy: 71.77%
    Epoch 5/10: Train Loss: 0.6997, Train Accuracy: 75.38%, Test Loss: 0.8028, Test Accuracy: 72.99%
    Epoch 6/10: Train Loss: 0.6086, Train Accuracy: 78.83%, Test Loss: 0.8065, Test Accuracy: 72.91%
    Epoch 7/10: Train Loss: 0.5290, Train Accuracy: 81.78%, Test Loss: 0.7920, Test Accuracy: 73.87%
    Epoch 8/10: Train Loss: 0.4433, Train Accuracy: 84.49%, Test Loss: 0.8443, Test Accuracy: 73.31%
    Epoch 9/10: Train Loss: 0.3747, Train Accuracy: 86.91%, Test Loss: 0.8515, Test Accuracy: 75.20%
    Epoch 10/10: Train Loss: 0.3126, Train Accuracy: 89.00%, Test Loss: 0.8525, Test Accuracy: 74.54%
    Total time taken for training and testing: 151.19 seconds
    Running with ELU activation
    Epoch 1/10: Train Loss: 1.6000, Train Accuracy: 41.73%, Test Loss: 1.3051, Test Accuracy: 53.62%
    Epoch 2/10: Train Loss: 1.1131, Train Accuracy: 60.54%, Test Loss: 1.0528, Test Accuracy: 62.47%
    Epoch 3/10: Train Loss: 0.8960, Train Accuracy: 68.65%, Test Loss: 0.9562, Test Accuracy: 67.44%
    Epoch 4/10: Train Loss: 0.7710, Train Accuracy: 72.98%, Test Loss: 0.7960, Test Accuracy: 72.31%
    Epoch 5/10: Train Loss: 0.6540, Train Accuracy: 77.34%, Test Loss: 0.7497, Test Accuracy: 74.41%
    Epoch 6/10: Train Loss: 0.5588, Train Accuracy: 80.63%, Test Loss: 0.8036, Test Accuracy: 73.81%
    Epoch 7/10: Train Loss: 0.4663, Train Accuracy: 83.76%, Test Loss: 0.8054, Test Accuracy: 74.34%
    Epoch 8/10: Train Loss: 0.3850, Train Accuracy: 86.53%, Test Loss: 0.8055, Test Accuracy: 75.07%
    Epoch 9/10: Train Loss: 0.3058, Train Accuracy: 89.37%, Test Loss: 0.8819, Test Accuracy: 74.79%
    Epoch 10/10: Train Loss: 0.2513, Train Accuracy: 91.44%, Test Loss: 1.0102, Test Accuracy: 74.38%
    Total time taken for training and testing: 156.92 seconds
    Running with SELU activation
    Epoch 1/10: Train Loss: 1.6541, Train Accuracy: 40.84%, Test Loss: 1.3572, Test Accuracy: 50.76%
    Epoch 2/10: Train Loss: 1.1416, Train Accuracy: 59.50%, Test Loss: 1.0852, Test Accuracy: 62.17%
    Epoch 3/10: Train Loss: 0.9409, Train Accuracy: 66.86%, Test Loss: 1.0228, Test Accuracy: 65.13%
    Epoch 4/10: Train Loss: 0.8070, Train Accuracy: 71.54%, Test Loss: 0.8711, Test Accuracy: 69.76%
    Epoch 5/10: Train Loss: 0.7096, Train Accuracy: 75.23%, Test Loss: 0.9055, Test Accuracy: 70.49%
    Epoch 6/10: Train Loss: 0.6232, Train Accuracy: 78.18%, Test Loss: 0.7562, Test Accuracy: 74.18%
    Epoch 7/10: Train Loss: 0.5383, Train Accuracy: 81.32%, Test Loss: 0.8189, Test Accuracy: 73.53%
    Epoch 8/10: Train Loss: 0.4728, Train Accuracy: 83.56%, Test Loss: 0.7818, Test Accuracy: 75.53%
    Epoch 9/10: Train Loss: 0.3952, Train Accuracy: 86.16%, Test Loss: 0.8987, Test Accuracy: 73.73%
    Epoch 10/10: Train Loss: 0.3350, Train Accuracy: 88.34%, Test Loss: 0.8563, Test Accuracy: 75.29%
    Total time taken for training and testing: 152.71 seconds
    Running with Softplus activation
    Epoch 1/10: Train Loss: 1.8854, Train Accuracy: 34.93%, Test Loss: 1.5127, Test Accuracy: 44.74%
    Epoch 2/10: Train Loss: 1.2166, Train Accuracy: 56.79%, Test Loss: 1.1173, Test Accuracy: 60.46%
    Epoch 3/10: Train Loss: 0.9857, Train Accuracy: 65.13%, Test Loss: 1.0494, Test Accuracy: 64.43%
    Epoch 4/10: Train Loss: 0.8497, Train Accuracy: 70.10%, Test Loss: 0.9086, Test Accuracy: 69.11%
    Epoch 5/10: Train Loss: 0.7413, Train Accuracy: 74.02%, Test Loss: 0.8447, Test Accuracy: 70.95%
    Epoch 6/10: Train Loss: 0.6546, Train Accuracy: 76.98%, Test Loss: 0.8557, Test Accuracy: 70.71%
    Epoch 7/10: Train Loss: 0.5722, Train Accuracy: 79.72%, Test Loss: 0.7750, Test Accuracy: 74.69%
    Epoch 8/10: Train Loss: 0.4908, Train Accuracy: 82.66%, Test Loss: 0.8189, Test Accuracy: 73.30%
    Epoch 9/10: Train Loss: 0.4104, Train Accuracy: 85.48%, Test Loss: 0.8777, Test Accuracy: 74.23%
    Epoch 10/10: Train Loss: 0.3468, Train Accuracy: 87.63%, Test Loss: 0.8002, Test Accuracy: 76.09%
    Total time taken for training and testing: 154.65 seconds
    Running with Mish activation
    Epoch 1/10: Train Loss: 1.7003, Train Accuracy: 39.14%, Test Loss: 1.3325, Test Accuracy: 51.16%
    Epoch 2/10: Train Loss: 1.1959, Train Accuracy: 57.22%, Test Loss: 1.3133, Test Accuracy: 54.21%
    Epoch 3/10: Train Loss: 1.0379, Train Accuracy: 63.33%, Test Loss: 1.0978, Test Accuracy: 63.63%
    Epoch 4/10: Train Loss: 0.8866, Train Accuracy: 69.08%, Test Loss: 0.8707, Test Accuracy: 70.11%
    Epoch 5/10: Train Loss: 0.7194, Train Accuracy: 74.65%, Test Loss: 0.8252, Test Accuracy: 71.92%
    Epoch 6/10: Train Loss: 0.6115, Train Accuracy: 78.71%, Test Loss: 0.7833, Test Accuracy: 73.93%
    Epoch 7/10: Train Loss: 0.5023, Train Accuracy: 82.60%, Test Loss: 0.8402, Test Accuracy: 72.62%
    Epoch 8/10: Train Loss: 0.4273, Train Accuracy: 84.96%, Test Loss: 1.0618, Test Accuracy: 70.38%
    Epoch 9/10: Train Loss: 0.3230, Train Accuracy: 88.85%, Test Loss: 0.8752, Test Accuracy: 73.95%
    Epoch 10/10: Train Loss: 0.3492, Train Accuracy: 87.84%, Test Loss: 0.9737, Test Accuracy: 75.13%
    Total time taken for training and testing: 162.41 seconds
    Running with GELU activation
    Epoch 1/10: Train Loss: 1.7372, Train Accuracy: 37.46%, Test Loss: 1.3839, Test Accuracy: 49.43%
    Epoch 2/10: Train Loss: 1.1420, Train Accuracy: 59.31%, Test Loss: 1.0404, Test Accuracy: 64.37%
    Epoch 3/10: Train Loss: 0.9815, Train Accuracy: 65.16%, Test Loss: 1.0502, Test Accuracy: 62.14%
    Epoch 4/10: Train Loss: 0.8416, Train Accuracy: 70.36%, Test Loss: 0.9324, Test Accuracy: 68.07%
    Epoch 5/10: Train Loss: 0.7057, Train Accuracy: 75.38%, Test Loss: 0.8300, Test Accuracy: 71.26%
    Epoch 6/10: Train Loss: 0.6015, Train Accuracy: 79.01%, Test Loss: 0.7985, Test Accuracy: 73.44%
    Epoch 7/10: Train Loss: 0.5035, Train Accuracy: 82.51%, Test Loss: 0.7912, Test Accuracy: 73.74%
    Epoch 8/10: Train Loss: 0.4082, Train Accuracy: 85.73%, Test Loss: 0.8449, Test Accuracy: 74.16%
    Epoch 9/10: Train Loss: 0.3402, Train Accuracy: 88.30%, Test Loss: 0.8916, Test Accuracy: 73.97%
    Epoch 10/10: Train Loss: 0.2871, Train Accuracy: 89.97%, Test Loss: 0.9368, Test Accuracy: 73.92%
    Total time taken for training and testing: 150.29 seconds
    Running with Swish activation
    Epoch 1/10: Train Loss: 1.7318, Train Accuracy: 37.64%, Test Loss: 1.3979, Test Accuracy: 51.26%
    Epoch 2/10: Train Loss: 1.1694, Train Accuracy: 58.49%, Test Loss: 1.5716, Test Accuracy: 45.62%
    Epoch 3/10: Train Loss: 1.0223, Train Accuracy: 63.88%, Test Loss: 0.9294, Test Accuracy: 67.68%
    Epoch 4/10: Train Loss: 0.8995, Train Accuracy: 68.39%, Test Loss: 0.9304, Test Accuracy: 67.77%
    Epoch 5/10: Train Loss: 0.8207, Train Accuracy: 71.18%, Test Loss: 0.8368, Test Accuracy: 71.25%
    Epoch 6/10: Train Loss: 0.6815, Train Accuracy: 76.06%, Test Loss: 0.8100, Test Accuracy: 72.55%
    Epoch 7/10: Train Loss: 0.5747, Train Accuracy: 79.85%, Test Loss: 0.8057, Test Accuracy: 73.69%
    Epoch 8/10: Train Loss: 0.4812, Train Accuracy: 82.99%, Test Loss: 0.8506, Test Accuracy: 73.33%
    Epoch 9/10: Train Loss: 0.3622, Train Accuracy: 87.16%, Test Loss: 0.9636, Test Accuracy: 73.00%
    Epoch 10/10: Train Loss: 0.3020, Train Accuracy: 89.34%, Test Loss: 1.0346, Test Accuracy: 73.36%
    Total time taken for training and testing: 155.03 seconds


<img src="{{site.url}}/assets/img/activation-func-ablation.png" style="display: block; margin: auto;" />
    

    
