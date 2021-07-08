import pdbr
import os
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim import Adam

from asteroid.engine.optimizers import make_optimizer
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

from asteroid import DPRNNTasNet 
from asteroid.data import LibriMix
from asteroid.engine.system import System

from models.controller import Controller
import pytorch_lightning as pl
from system import AdvAutoAugment
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", default="exp/tmp", help="Full path to save best validation model")
pl.seed_everything(42)

def main(config):
    train_set = LibriMix(
        csv_dir=config["data"]["train_dir"],
        task=config["data"]["task"],
        sample_rate=config["data"]["sample_rate"],
        n_src=config["data"]["nondefault_nsrc"],
        segment=config["data"]["segment"]
    )


    val_set = LibriMix(
        csv_dir=config["data"]["valid_dir"],
        task=config["data"]["task"],
        sample_rate=config["data"]["sample_rate"],
        n_src=config["data"]["nondefault_nsrc"],
        segment=config["data"]["segment"],
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        drop_last=True,
    )
    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None

    target_model = DPRNNTasNet(**config["filterbank"], **config["masknet"])
    controller_model = Controller()

    target_model_optimizer =  make_optimizer(target_model.parameters(), **config["optim"])
    controller_model_optimizer = Adam(controller_model.parameters(), lr = 0.00035)

    loss_function = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    adv_autoaugment_model = AdvAutoAugment(target_model=target_model, 
                                           controller_model=controller_model,
                                           train_loader=train_loader,
                                           val_loader=val_loader,
                                           loss_function=loss_function,
                                           target_model_optimizer=target_model_optimizer,
                                           controller_model_optimizer=controller_model_optimizer,
                                           config=config,
                                           )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        gpus=gpus,
        distributed_backend="ddp",
        gradient_clip_val=config["training"]["gradient_clipping"],
    )
    trainer.fit(adv_autoaugment_model)



if __name__ == "__main__":
    import yaml
    from pprint import pprint as print
    from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    print(arg_dic)
    main(arg_dic)
