import random
import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import torch.distributed as dist
import copy
import yaml
import re
from collections import defaultdict

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_
    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.
    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1. / batch_size))
    
    return res
    
class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon = 0, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = (-targets * log_probs)
                
        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class Logger:
    def __init__(self,log = './logs'):
        self.metrics = defaultdict(lambda:[])
        self.log = log
        
        os.makedirs(self.log,exist_ok=True)
        os.makedirs(os.path.join(self.log,'models/'),exist_ok=True)
        
    def add(self, key, value):
        self.metrics[key].append(value)

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))
    
    def save_model(self,model, epoch):
        save_path = os.path.join(self.log,'models/')
        torch.save(model.module.state_dict(), os.path.join(save_path,'checkpoint_%d.pth'%epoch))
        if self.metrics['valid_loss'][-1] == np.min(self.metrics['valid_loss']):
            torch.save(model.module.state_dict(), os.path.join(save_path,'best_loss.pth'))
        if self.metrics['valid_top1'][-1] == np.max(self.metrics['valid_top1']):
            torch.save(model.module.state_dict(), os.path.join(save_path,'best_top1.pth'))
    
    def save_logs(self):
        save_path = os.path.join(self.log,'logs.pkl')        
        open_file = open(save_path, "wb")
        pickle.dump(self.get_dict(), open_file)
        open_file.close()
    
    def info(self, epoch, keys):
        output = "Epoch: {:d} ".format(epoch)
        for key in keys:
            output += "{}: {:.5f} ".format(key, self.metrics[key][-1])
        print(output)
        
def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt

def reduced_metric(metric,num_gpus,ddp=True):
    if ddp:
        reduced_loss = reduce_tensor(metric.data, num_gpus)
        return reduced_loss.item()
    return metric.item()

def load_yaml(dir):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    conf = yaml.load(open(dir, 'r'), Loader=loader)
    return conf
