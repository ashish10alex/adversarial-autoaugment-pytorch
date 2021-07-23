import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_lightning.core import LightningModule

class AugmentationProbabilityModel(LightningModule):
    def __init__(self, num_augmentations, embedding_dim):
        super(AugmentationProbabilityModel, self).__init__()
        self.num_augmentations = num_augmentations
        self.embedding_dim =  embedding_dim
        # needs better initialization
        # self.embedding = Variable(torch.zeros(self.num_augmentations, self.embedding_dim,  device='cuda'), requires_grad = False,)
        self.augmentation_indicies = torch.LongTensor(list(range(num_augmentations))).cuda()
        self.embedding_layer = nn.Embedding(self.num_augmentations, self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim * self.num_augmentations, self.num_augmentations)
        self.softmax = nn.Softmax()


    def forward(self):
        embeddings = self.embedding_layer(self.augmentation_indicies) #torch.Size([4, 20])
        output = embeddings.reshape(self.embedding_dim * self.num_augmentations)
        output = self.linear(output)
        probabilities = self.softmax(output)
        return probabilities
