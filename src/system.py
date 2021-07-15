
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable
from asteroid import DPRNNTasNet
from pytorch_lightning.core import LightningModule
from torch.optim import Adam, SGD
from collections import OrderedDict

NUM_OPS = 15 # NUM_OPS is the Number of image operations in the search space. 16 in paper
NUM_MAGS = 10 # Maginitde of the operations discrete 10 values

        
class AdvAutoAugment(LightningModule):
    def __init__(self, target_model, controller_model, loss_function, train_loader, val_loader, target_model_optimizer,
                 controller_model_optimizer,  config):
        super().__init__()
        self.target_model = target_model
        self.controller_model = controller_model
        self.lr = config['optim']['lr']
        self.config = config
        self.weight_decay = self.config['optim']['weight_decay']
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_model_optimizer = target_model_optimizer
        self.controller_model_optimizer = controller_model_optimizer
        self.config = config

    def forward(self, mixed_audio_batch):
        return self.target_model(mixed_audio_batch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        #Train target model
        if optimizer_idx == 0:
            mixture, sources = batch
            #TODO - 
            #1. Create multiple instances of batch 
            estimated_sources = self(mixture)
            self.target_model_loss = self.loss_function(estimated_sources, sources)
            self.target_model_loss_copy = self.target_model_loss.clone().detach() # is this hack correct
            # print(f'self.target_model_loss_copy: {self.target_model_loss_copy}')
            target_model_output = OrderedDict({
                                'loss': self.target_model_loss,
            })
            self.log('target_loss', self.target_model_loss_copy, prog_bar=True)
            return target_model_output

        if optimizer_idx == 1:
            # Add controller logic later
            policies, log_probs, entropies = self.controller_model(self.config["controller"]["M"]) # (M,2*2*5) (M,) (M,) 
            # normalized_target_model_loss = (self.target_model_loss - torch.mean(self.target_model_loss))/(torch.std(self.target_model_loss) + 1e-5)
            normalized_target_model_loss = self.target_model_loss_copy
            # normalized_target_model_loss = torch.Tensor([10.0]).cuda()
            score_loss = torch.mean(-log_probs * normalized_target_model_loss) # - derivative of Score function
            entropy_penalty = torch.mean(entropies) # Entropy penalty
            controller_loss = score_loss - self.config['controller']['entropy_penalty'] * entropy_penalty
            # print(f'controller_loss: {controller_loss}')
            controller_model_output = OrderedDict({
                                'loss': controller_loss,
            })
            self.log('controller_loss', controller_loss, prog_bar=True)
            return controller_model_output

    # def backward(self, loss, optimizer, optimizer_idx):
    #     if optimizer_idx == 0:
    #         # print(f'optmizer idx: {optimizer_idx}, loss: {loss}')
    #         loss.backward()
    #     if optimizer_idx == 1:
    #         # print(f'optmizer idx: {optimizer_idx}, loss: {loss}')
    #         loss.backward()

    def validation_step(self, batch, batch_nb):
        mixture, sources = batch
        estimated_sources = self(mixture)
        self.target_model_loss = self.loss_function(estimated_sources, sources)
        self.log("val_loss", self.target_model_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        '''
        Add scheduler ?
        '''
        target_model_optimizer = self.target_model_optimizer
        controller_model_optimizer = self.controller_model_optimizer 
        return [target_model_optimizer, controller_model_optimizer], []

    def on_epoch_end(self):
        pass


    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader



