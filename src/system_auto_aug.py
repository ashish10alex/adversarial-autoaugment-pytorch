import time
from collections import OrderedDict

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from augment_with_policies import (AugmentWithPolicies,
                                   ComposeAugmentationPolices,
                                   augmentation_list)
from utils import flatten_dict


class SystemAutoAug(pl.LightningModule):
    default_monitor: str = "val_loss"

    def __init__(self, target_model, probability_model, loss_function, train_loader, val_loader, target_model_optimizer,
                 probability_model_optimizer, scheduler, config):
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
        self.scheduler = scheduler
        self.automatic_optimization = False # to do manual training step
        # self.save_hyperparameters(self.config_to_hparams(self.config))

    def get_augmented_data_with_policies(self, mixture, sources, probabilities):
        augmentation_policies = ComposeAugmentationPolices(probabilities, augmentation_list=augmentation_list)
        augmentation_function = AugmentWithPolicies(augmentation_policies())
        mixture = augmentation_function(mixture.detach().cpu().numpy())
        mixture = torch.Tensor(mixture).cuda()
        # mixture = torch.Tensor(mixture)
        sources = einops.repeat(sources, 'b h w -> (repeat b) h w', repeat=mixture.shape[0])
        return mixture, sources

    def get_probability_model_loss(self, probabilities, target_model_loss):
        '''
        Loss function seems to have the same effect on all the probabilities
        '''
        prob_loss = [p*target_model_loss for p in probabilities]
        prob_loss = sum(prob_loss)/len(prob_loss)
        return prob_loss

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.target_model(*args, **kwargs)

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

    def validation_step(self, batch, batch_nb):
        mixture, sources = batch
        estimated_sources = self.target_model(mixture)
        self.target_model_loss = self.loss_function(estimated_sources, sources)
        self.log("val_loss", self.target_model_loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        """Log hp_metric to tensorboard for hparams selection."""
        hp_metric = self.trainer.callback_metrics.get("val_loss", None)
        if hp_metric is not None:
            self.trainer.logger.log_metrics({"hp_metric": hp_metric}, step=self.trainer.global_step)

    def configure_optimizers(self):
        '''
        Add scheduler ?
        '''
        target_model_optimizer = self.target_model_optimizer
        probability_model_optimizer = self.probability_model_optimizer 
        epoch_schedulers = []
        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler] 
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [target_model_optimizer, probability_model_optimizer], epoch_schedulers

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
