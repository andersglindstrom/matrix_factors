#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

A = torch.rand(5, 2)
B = torch.rand(4, 2)

print(f"A:\n{A}")
print(f"B:\n{B}")

AB = A@B.T

print(f"AB:\n{AB}")

# Want use gradient descent to find B

class MatrixFactorisation(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(4, 2))

    def forward(self, x):
        return x@self.weights.T


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
# device = "cuda"
print(f"Device={device}")

model = MatrixFactorisation().to(device)
print(f"Initial model.weights:\n{model.weights}")

loss_fn = nn.MSELoss()
# loss_fn = nn.HuberLoss(delta=1)

# optimizer = optim.Adam(model.parameters(), lr=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# optimizer = optim.SGD(model.parameters(), lr=0.1)

num_epochs = 1000

A = A.to(device)
AB = AB.to(device)

for epoch in range(num_epochs):

    optimizer.zero_grad()

    # Forward pass
    output = model.forward(A)
    loss = loss_fn(output, AB)
    # print(f"loss: {loss}")

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()


print(f"Target:\n{B}")
print(f"Final model.weights:\n{model.weights}")
