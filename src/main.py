import argparse
import numpy as np
import torch
import pdbr

import sys
from tqdm import tqdm
from dataloader.dataloader import get_dataloader
from torch.cuda.amp import GradScaler, autocast
from dataloader.transform import parse_policies, MultiAugmentation
from optimizer_scheduler import get_optimizer_scheduler
from models import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='Unofficial Implementation of Adversarial Autoaugment')
    parser.add_argument('--load_conf', type = str, default='confs/cifar10/shake26_2x96d_cifar10.yaml')
    parser.add_argument('--logdir', type = str, default='./logs')
    # M is instances of input example augmented by adverserial policies
    parser.add_argument('--M', type = int, default=8)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--local_rank', type = int, default = -1)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    return args

def init_ddp(local_rank):
    if local_rank !=-1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method='env://')

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    
    conf = load_yaml(args.load_conf)
    logger = Logger(os.path.join(args.logdir,args.load_conf.split('/')[-1].split('.')[0] + "_%d"%args.seed))
    num_gpus = torch.cuda.device_count()    
    
    ## DDP set print_option + initialization
    if args.local_rank > 0:
        sys.stdout = open(os.devnull, 'w')
    init_ddp(args.local_rank)
    print("EXPERIMENT:",args.load_conf.split('/')[-1].split('.')[0])
    print()
    
    train_sampler, train_loader, valid_loader, test_loader = get_dataloader(conf, dataroot = './dataloader/datasets', split = 0, split_idx = 0, multinode = (args.local_rank!=-1))    
    
    controller = get_controller(conf,args.local_rank)
    model = get_model(conf,args.local_rank)
    (optimizer, scheduler), controller_optimizer = get_optimizer_scheduler(controller, model, conf)
    criterion = CrossEntropyLabelSmooth(num_classes = num_class(conf['dataset']))
    
    if args.amp:
        scaler = GradScaler()
    
    step = 0
    for epoch in range(conf['epoch']):
        if args.local_rank >=0:
            train_sampler.set_epoch(epoch)
        
        Lm = torch.zeros(args.M).cuda() #used to store loss [M]
        Lm.requires_grad = False
        
        model.train() #image recognition model
        controller.train() #policy network
        #Batch size in controller is args.M (default 5) which defines the number of policies.
        policies, log_probs, entropies = controller(args.M) # (M,2*2*5) (M,) (M,) 
        policies = policies.cpu().detach().numpy()
        # list of augmentations [M, ]
        parsed_policies = parse_policies(policies)
        
        trfs_list = train_loader.dataset.dataset.transform.transforms 
        trfs_list[2] = MultiAugmentation(parsed_policies)## replace augmentation into new one
        
        train_loss = 0
        train_top1 = 0
        train_top5 = 0
        
        progress_bar = tqdm(train_loader)
        for idx, (data,label) in enumerate(progress_bar):
            # print(f"batch number: {idx}")
            optimizer.zero_grad()
            data = data.cuda()
            label = label.cuda()
            #arg.amp = store_true
            with autocast(enabled=args.amp):
                pred = model(data)
                losses = [criterion(pred[i::args.M,...] ,label) for i in range(args.M)]
                loss = torch.mean(torch.stack(losses))
            
            if args.amp:
                #Enabled when using automatic mixed precision to prevent `underflow` of gradients.
                # read - https://pytorch.org/docs/stable/amp.html
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            for i,_loss in enumerate(losses):
                # Why divide by `len(train_loader)` which is = 8333
                Lm[i] += reduced_metric(_loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader) #default 
                # Lm[i] += reduced_metric(_loss.detach(), num_gpus, args.local_rank !=-1)  / train_loader.batch_size * args.M #changed
            
            top1 = None
            top5 = None
            for i in range(args.M):
                _top1,_top5 = accuracy(pred[i::args.M,...], label, (1, 5))
                top1 = top1 + _top1/args.M if top1 is not None else _top1/args.M
                top5 = top5 + _top5/args.M if top5 is not None else _top5/args.M
            
            train_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            train_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) / len(train_loader)
            
            progress_bar.set_description('Step: {}. LR : {:.5f}. Epoch: {}/{}. Iteration: {}/{}. Train_Loss : {:.5f}'.format(step, optimizer.param_groups[0]['lr'], epoch, conf['epoch'], idx + 1, len(train_loader), loss.item()))
            step += 1

        model.eval()
        controller.train()
        controller_optimizer.zero_grad()
        
        normalized_Lm = (Lm - torch.mean(Lm))/(torch.std(Lm) + 1e-5)
        score_loss = torch.mean(-log_probs * normalized_Lm) # - derivative of Score function
        entropy_penalty = torch.mean(entropies) # Entropy penalty
        controller_loss = score_loss - conf['entropy_penalty'] * entropy_penalty
        controller_loss.backward()
        controller_optimizer.step()
        scheduler.step()
        
        valid_loss = 0.
        valid_top1 = 0.
        valid_top5 = 0.
        cnt = 0.
        with torch.no_grad():
            for idx, (data,label) in enumerate(tqdm(test_loader)):
                b = data.size(0)
                data = data.cuda()
                label = label.cuda()
                
                pred = model(data)
                loss = criterion(pred,label)

                top1, top5 = accuracy(pred, label, (1, 5))
                valid_loss += reduced_metric(loss.detach(), num_gpus, args.local_rank !=-1) *b 
                valid_top1 += reduced_metric(top1.detach(), num_gpus, args.local_rank !=-1) *b
                valid_top5 += reduced_metric(top5.detach(), num_gpus, args.local_rank !=-1) *b 
                cnt += b
            
            valid_loss = valid_loss / cnt
            valid_top1 = valid_top1 / cnt
            valid_top5 = valid_top5 / cnt
            
        logger.add_dict(
            {
                'train_loss' : train_loss,
                'train_top1' : train_top1,
                'train_top5' : train_top5,
                'controller_loss' : controller_loss.item(),
                'score_loss' : score_loss.item(),
                'entropy_penalty' : entropy_penalty.item(),
                'valid_loss' : valid_loss,
                'valid_top1' : valid_top1,
                'valid_top5' : valid_top5,
                'policies' : parsed_policies,
            }
        )
        if args.local_rank <= 0:
            logger.save_model(model,epoch)
        logger.info(epoch,['train_loss','train_top1','train_top5','valid_loss','valid_top1','valid_top5','controller_loss'])
    
        logger.save_logs()
