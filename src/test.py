from tqdm import tqdm
import pdbr
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from tqdm import trange


class ProbModel(nn.Module):
    def __init__(self, input_size=3):
        super(ProbModel, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(self.input_size, self.input_size)
        self.sigmoid = nn.Sigmoid()
        self._input  = torch.randn(1, input_size, device='cuda')

    def forward(self):
        out = self.linear(self._input)
        out = self.sigmoid(out)
        return out[0]

if __name__ == '__main__':
    input_size = 3
    model = ProbModel(input_size=input_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    _input = torch.randn(1, input_size)
    print(f'_input:{_input}')
# ground_truth = torch.randn(1, 3)
    ground_truth = torch.Tensor([0.5, 0.3, 0.2])
    print(f'ground_truth:{ground_truth}')
# ground_truth = _input

    loss_function = nn.L1Loss()

    items = list(range(10000))
    pbar = tqdm(items)

    for i in pbar:
        opt.zero_grad()
        output = model()
        loss = loss_function(output, ground_truth)
        loss.backward()
        opt.step()
        pbar.set_postfix({'loss': loss.item(), 
                          'pred1': output[0][0].item(), 
                          'pred2': output[0][1].item(), 
                          'pred3': output[0][2].item(),
                          })
        time.sleep(0.1)

