import pdbr
import torch
import torch.nn as nn
from torch.autograd import Variable

model = nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 3),
    nn.Sigmoid()
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

_input = torch.randn(1, 10)
ground_truth = torch.randn(1, 3)

loss_function = nn.L1Loss()

for i in range(9000):
    opt.zero_grad()

    output = model(_input)
    print(output)
    loss = loss_function(output, ground_truth)
    print(loss)

    loss.backward()
    opt.step()
