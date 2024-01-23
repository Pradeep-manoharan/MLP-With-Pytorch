import numpy
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch.nn as nn

# Data Preparation


x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(x_numpy.astype(numpy.float32))
Y = torch.from_numpy(y_numpy.astype(numpy.float32))

Y = Y.view(Y.shape[0], -1)

X_sample, n_feature = X.shape

input_size = n_feature
output_size = 1

# Model Building

model = nn.Linear(input_size, output_size)

# Loss and optimizer

Learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

# Training Loop

num_epochs = 100

for epoch in range(num_epochs):
    # forward and loss
    y_predicted = model(X)

    loss = criterion(Y, y_predicted)

    # Backward pass
    loss.backward()

    # opimizer
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) % 10 ==0:
        print(f"epoch {epoch}, Loss ={loss.item():4f}")

#plot

predicted = model(X).detach().numpy()
plt.plot(x_numpy,y_numpy, "ro")
plt.plot(x_numpy,predicted,"b")
plt.show()
