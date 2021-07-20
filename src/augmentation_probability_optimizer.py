import torch
import torch.nn as nn
from torch.autograd import Variable

class AugmentationProbabilityOptimizer(nn.Module):
    def __init__(self, num_augmentations, embedding_dim):
        super(AugmentationProbabilityOptimizer, self).__init__()
        self.num_augmentations = num_augmentations
        self.embedding_dim =  embedding_dim
        self.embedding = Variable(torch.zeros(self.num_augmentations, self.embedding_dim), requires_grad = False) #better initialization
        self.linear = nn.Linear(self.embedding_dim, self.num_augmentations)

    def forward(self):
        output = self.linear(self.embedding)
        probabilities = torch.softmax(output, dim=1)
        return probabilities[0]
