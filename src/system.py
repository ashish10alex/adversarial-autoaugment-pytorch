
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from asteroid import DPRNNTasNet
from pytorch_lightning.core import LightningModule
from torch.optim import Adam, SGD

NUM_OPS = 15 # NUM_OPS is the Number of image operations in the search space. 16 in paper
NUM_MAGS = 10 # Maginitde of the operations discrete 10 values

        
class AdvAutoAugment(LightningModule):
    def __init__(self, target_model, controller_model, loss_function, train_loader, val_loader, config):
        super().__init__()
        self.target_model = target_model
        self.controller_model = controller_model
        self.lr = config['optim']['lr']
        self.config = config
        self.weight_decay = self.config['optim']['weight_decay']
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        

    def forward(mixed_audio_batch):
        return self.target_model(mixed_audio_batch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        breakpoint()
        print('training')
        return -1

    def configure_optimizers(self):
        '''
        Add scheduler ?
        '''
        controller_optimizer = Adam(self.controller_model.parameters(), lr = 0.00035)
        target_optimizer = SGD(self.target_model.parameters(), lr = self.lr, momentum = 0.9, nesterov = True,
                               weight_decay = self.weight_decay)
        return [controller_optimizer, target_optimizer], []

    def on_epoch_end(self):
        pass


    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader



