from tqdm import tqdm
import pdbr
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import trange

inp_size = 6
model = nn.Sequential(
    nn.Linear(inp_size, inp_size),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

_input = torch.randn(1, inp_size)
print(f'_input:{_input}')
# ground_truth = torch.randn(1, 3)
ground_truth = _input

loss_function = nn.L1Loss()

items = list(range(10000))
pbar = tqdm(items)

for i in pbar:
    opt.zero_grad()
    output = model(_input)
    loss = loss_function(output, ground_truth)
    loss.backward()
    opt.step()
    pbar.set_postfix({'loss': loss.item(), 
                      # 'pred1': output[0][0].item(), 
                      # 'pred2': output[0][1].item(), 
                      # 'pred3': output[0][2].item(),
                      # 'pred4': output[0][3].item(),
                      # 'pred5': output[0][4].item(),
                      # 'pred6': output[0][5].item(),
                      })
    time.sleep(0.1)

