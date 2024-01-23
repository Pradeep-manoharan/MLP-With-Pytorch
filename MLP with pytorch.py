import torch
import torchvision.datasets
from torchvision import datasets
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import torch.nn as nn

# Device Configuration

Device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Hyper Parameters

inputs_size = 784  # 28*28
hidden_size = 500
batch_size = 100
num_classes = 10
learning_rate = 0.01
num_epochs = 2

# MNIST Dataset

train_dataset = torchvision.datasets.MNIST("\data",
                                           train=True,
                                           download=True,
                                           transform=transform.ToTensor())

test_dataset = torchvision.datasets.MNIST('\data',
                                          train=True,
                                          transform=transform.ToTensor())

# Data Loader

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

example = iter(test_loader)

example_image, example_target = next(example)


# plt.figure(figsize=(5, 4))
# for i in range(6):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(example_image[i][0], cmap="gray")
#     plt.title(example_target[i])
#
# plt.show()

# Fully connected the network with one hidden layer

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size,bias=True)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes,bias=True)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


model = NeuralNetwork(inputs_size, hidden_size, num_classes).to(Device)

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):

        image = image.reshape(-1, 28 * 28).to(Device)
        label = label.to(Device)

        # Forward pass

        outputs = model(image)
        loss = criterion(outputs, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epochs [{epoch + 1} /{num_epochs}],step [{i + 1}/{n_total_step}], loss: {loss.item():.4f}')
