import einops
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
from augment_with_policies import augmentation_list, ComposeAugmentationPolices, AugmentWithPolicies

NUM_OPS = 15 # NUM_OPS is the Number of image operations in the search space. 16 in paper
NUM_MAGS = 10 # Maginitde of the operations discrete 10 values

        
class AdvAutoAugment(LightningModule):
    def __init__(self, target_model, probability_model, loss_function, train_loader, val_loader, target_model_optimizer,
                 probability_model_optimizer,  config):
        super().__init__()
        self.target_model = target_model
        self.probability_model = probability_model
        self.lr = config['optim']['lr']
        self.config = config
        self.weight_decay = self.config['optim']['weight_decay']
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_model_optimizer = target_model_optimizer
        self.probability_model_optimizer = probability_model_optimizer
        self.config = config

    def forward(self, mixed_audio_batch):
        return self.target_model(mixed_audio_batch)

    def get_augmented_data_with_policies(self, mixture, sources, probabilities):
        augmentation_policies = ComposeAugmentationPolices(probabilities, augmentation_list=augmentation_list)
        augmentation_function = AugmentWithPolicies(augmentation_policies())
        mixture = augmentation_function(mixture.detach().cpu().numpy()[0])
        # mixture = torch.Tensor(mixture).cuda()
        mixture = torch.Tensor(mixture)
        sources = einops.repeat(sources, 'b h w -> (repeat b) h w', repeat=mixture.shape[0])
        return mixture, sources


    def training_step(self, batch, batch_idx, optimizer_idx):
        #Train target model
        if optimizer_idx == 0:
            mixture, sources = batch
            probabilities = self.probability_model() # (M,2*2*5) (M,) (M,) 
            print(f'probabilities: {probabilities}')
            mixture, sources = self.get_augmented_data_with_policies(mixture, sources, probabilities)
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

        # if optimizer_idx == 1:
        #     # Add probability logic later
        #     probabilities = self.probability_model() # (M,2*2*5) (M,) (M,) 
        #     # normalized_target_model_loss = (self.target_model_loss - torch.mean(self.target_model_loss))/(torch.std(self.target_model_loss) + 1e-5)
        #     probability_loss = self.target_model_loss_copy 
        #     # normalized_target_model_loss = torch.Tensor([10.0]).cuda()
        #     probability_model_output = OrderedDict({
        #                         'loss': probability_loss,
        #     })
        #     self.log('probability_loss', probability_loss, prog_bar=True)
        #     return probability_model_output

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
        probability_model_optimizer = self.probability_model_optimizer 
        return [target_model_optimizer, probability_model_optimizer], []

    def on_epoch_end(self):
        pass


    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader



