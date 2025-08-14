import logging

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)

class Model(torch.nn.Module):
    def __init__(self, input_=784, hidden_1=1024, hidden_2=1024, output=10):
        super().__init__()

        self.flatten = torch.nn.Flatten()
        self.layer1 = torch.nn.Linear(input_, hidden_1)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.layer2 = torch.nn.Linear(hidden_1, hidden_2)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.layer3 = torch.nn.Linear(hidden_2, output)

    def forward(self, data):
        data = self.flatten(data)
        data = torch.relu(self.layer1(data))
        data = self.dropout1(data)
        data = torch.relu(self.layer2(data))
        data = self.dropout2(data)
        data = self.layer3(data)

        return data

model = Model().to(device)

dataset_mnist = torchvision.datasets.MNIST(
    root='MNIST',
    train=True,
    download=False,
    transform=torchvision.transforms.ToTensor()
)

train_X = []
test_X = []

train_y = []
test_y = []

for i in range(10000):
    train_X.append(dataset_mnist[i][0])
    train_y.append(dataset_mnist[i][1])

for i in range(4001, 5001):
    test_X.append(dataset_mnist[i][0])
    test_y.append(dataset_mnist[i][1])

array_train_X = np.array(train_X)
array_test_X = np.array(test_X)

#final dataset
tensor_train_X = torch.tensor(array_train_X)
tensor_test_X = torch.tensor(array_test_X)

tensor_train_y = torch.tensor(train_y)
tensor_test_y = torch.tensor(test_y)

tensor_train_X = tensor_train_X.to(device)
tensor_test_X = tensor_test_X.to(device)

tensor_train_y = tensor_train_y.to(device)
tensor_test_y = tensor_test_y.to(device)

train_dataset = torch.utils.data.TensorDataset(tensor_train_X, tensor_train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
figure, axis = plt.subplots()

losses_list = []
epochs_list = []


def train_model():
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            prediction = model.forward(batch_X)

            loss = criterion(prediction, batch_y)

            losses_list.append(loss.item())
            epochs_list.append(epoch)

            loss.backward()
            optimizer.step()

    axis.plot(epochs_list, losses_list)

    axis.set_xlabel('epoch')
    axis.set_ylabel('loss')

    plt.show()

    losses_list.clear()
    epochs_list.clear()


def test_model():
    prediction = model.forward(tensor_test_X)

    loss = criterion(prediction, tensor_test_y)

    logging.info(f'Потерь модели на тестовом датасете: {loss}')


train_model()
test_model()
