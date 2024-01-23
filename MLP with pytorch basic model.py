import torch
import torch.nn as nn

# Design model (input, output size, forward pass)
# Construct Loss and optimizer
# Training Loop
# -->Forward pass : compute prediction
# --> backward pass : graditents
# --->update weight

# Inputs

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction

def forward(x):
    return w * x


print(f'Prediction before training : f(5) = {forward(5):.3f}')

# Training

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)

for epochs in range(n_iters):
    # prediction = forward pass

    y_pred = forward(X)

    # loss
    l = loss(Y,y_pred)

    # gradients = backward pass
    l.backward()

    # update weights

    optimizer.step()

    # zero gradients

    w.grad.zero_()

    if epochs % 2 == 0:
        print(f'epoch {epochs+1}: w = {w:.3f}, loss = {l:.8f}')







