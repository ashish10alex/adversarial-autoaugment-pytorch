import time
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
        self.automatic_optimization = False # to do manual training step

    def forward(self, mixed_audio_batch):
        return self.target_model(mixed_audio_batch)

    def get_augmented_data_with_policies(self, mixture, sources, probabilities):
        augmentation_policies = ComposeAugmentationPolices(probabilities, augmentation_list=augmentation_list)
        augmentation_function = AugmentWithPolicies(augmentation_policies())
        mixture = augmentation_function(mixture.detach().cpu().numpy())
        mixture = torch.Tensor(mixture).cuda()
        # mixture = torch.Tensor(mixture)
        sources = einops.repeat(sources, 'b h w -> (repeat b) h w', repeat=mixture.shape[0])
        return mixture, sources

    def test_how_many_audios_augemented(self, mixture):
        is_equal = []
        half = mixture.size()[0] // 2
        for i in range(half):
            is_equal.append(torch.equal(mixture[0], mixture[i]))
        return is_equal

    def training_step(self, batch, batch_idx):
        #DO NOT AUGMENT IN EVERY ITERATION
        optimizer_1, optimizer_2 = self.optimizers()
        mixture, sources = batch
        probabilities = self.probability_model() 
        probabilities_copy = probabilities.clone().detach() #make a copy so that they point to different address in memory
        # probabilities_copy = torch.Tensor([0.0, 0.2, 0.2, 0.2]).cuda()
        #Print statements to check if the gradients are updating
        # print(f'self.probability_model.linear.weight: {self.probability_model.linear.weight}')
        # print(f'probabilities: {probabilities}')
        # print(f'grads prob_model: {list(self.probability_model.parameters())[0].grad is not None}') #should be true if working
        # print(f'grads target_model: {list(self.target_model.parameters())[0].grad is not None}') #should be true if working
        # mixture1, sources1 = self.get_augmented_data_with_policies(mixture[0], sources[0][None], probabilities_copy)
        # mixture2, sources2 = self.get_augmented_data_with_policies(mixture[1], sources[1][None], probabilities_copy)
        mixture, sources = self.get_augmented_data_with_policies(mixture[0], sources[0][None], probabilities_copy)
        # mixture = torch.cat([mixture1, mixture2], dim=0)
        # sources = torch.cat([sources1, sources2], dim=0)

        #Tests to check if audio is augmented all the time
        # print(f'mixtures equal: {self.test_how_many_audios_augemented(mixture)}')
        # print(f'sources equal: {self.test_how_many_audios_augemented(sources)}')
        estimated_sources = self.target_model(mixture)
        target_model_loss = self.loss_function(estimated_sources, sources)
        optimizer_1.zero_grad()
        # self.manual_backward(target_model_loss, optimizer_1, retain_graph=True)
        # target_model_loss.backward(retain_graph=True)
        target_model_loss.backward()
        optimizer_1.step()

        optimizer_2.zero_grad()
        loss_prob_fake = self.get_probability_model_loss(probabilities, target_model_loss.detach())
        loss_prob_fake.backward(inputs=list(self.probability_model.parameters()))
        # self.manual_backward(target_model_loss, optimizer_2, inputs=list(self.probability_model.parameters()))
        optimizer_2.step()
        target_model_output = OrderedDict({
                            'target_model_loss': target_model_loss,
        })
        self.log('target_loss', target_model_loss, prog_bar=True)
        self.log('prob1', probabilities_copy[0], prog_bar=True)
        self.log('prob2', probabilities_copy[1], prog_bar=True)
        self.log('prob3', probabilities_copy[2], prog_bar=True)
        self.log('prob4', probabilities_copy[3], prog_bar=True)

        # self.trainer.logger.log_metrics(target_model_output)
        # self.log('prob1', probabilities_copy[0])
        return target_model_output

    def get_probability_model_loss(self, probabilities, target_model_loss):
        '''
        Loss function seems to have the same effect on all the probabilities
        '''
        prob_loss = [p*target_model_loss for p in probabilities]
        prob_loss = sum(prob_loss)/len(prob_loss)
        return prob_loss

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



