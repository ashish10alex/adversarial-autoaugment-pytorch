import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

NUM_OPS = 15 # NUM_OPS is the Number of image operations in the search space. 16 in paper
NUM_MAGS = 10 # Maginitde of the operations discrete 10 values

class Controller(nn.Module):
    def __init__(self,n_subpolicies = 5, embedding_dim = 32,hidden_dim = 100):
        super(Controller, self).__init__()
        self.Q = n_subpolicies
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(NUM_OPS + NUM_MAGS,embedding_dim) # (# of operation) + (# of magnitude) 
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.outop = nn.Linear(hidden_dim,NUM_OPS)
        self.outmag = nn.Linear(hidden_dim,NUM_MAGS)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.outop.bias.data.fill_(0)
        self.outmag.bias.data.fill_(0)
        
    def get_variable(self, inputs, cuda=False, **kwargs):
        '''
        Converts to torch tensor on appropriate device and wraps it in the Variable class
        '''
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
    
    def create_static(self,batch_size):
        inp = self.get_variable(torch.zeros(batch_size, self.embedding_dim), cuda = True, requires_grad = False)
        hx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = True, requires_grad = False)
        cx = self.get_variable(torch.zeros(batch_size, self.hidden_dim), cuda = True, requires_grad = False)
        
        return inp,hx,cx
    
    def calculate(self,logits):
        #All logits seem to be the same i.e [logits[i] == logits[i+1 ... n]]
        #Logits is the unnormalized final scores of the model. Softmax is applied to get probility 
        # distribution over classes
        probs = F.softmax(logits, dim=-1) #(M, NUM_OPS) 
        # log softmax heavily penalize stuff that fails to predict the correct class
        # Log softmax has also got improved numerical performance and gradient optimization
        log_prob = F.log_softmax(logits, dim=-1) 
        # Entropy measures the information or uncertainity of the variable.
        # Entropy is measured in bits and there can be more than one bit of
        # information in a varibale 
        entropy = -(log_prob * probs).sum(1, keepdim=False)
        #Action seems to be randomly selected 8 indices in [0, NUM_OPS)
        action = probs.multinomial(num_samples=1).data #[M, 1]
        #selected_log_prob is log probability value corrosponsing to the index value 
        selected_log_prob = log_prob.gather(1, self.get_variable(action,requires_grad = False))
        #Resize seleted_log_prob and action from torch.Size([M, 1]) to torch.size([M])
        return entropy, selected_log_prob[:, 0], action[:,0]
    
    def forward(self,batch_size=1):
        #batch size is number of policies not to be confused with batch size used to train the dataset
        #batch size is passed as args.M which defines the number of policies.
        return self.sample(batch_size)
    
    def sample(self,batch_size=1):
        policies = []
        entropies = []
        log_probs = []
           
#        inp,hx,cx = self.create_static(batch_size)
        #Q is number of policies
        for i in range(self.Q):
            inp,hx,cx = self.create_static(batch_size)
            for j in range(2):
#                if i > 0 or j > 0:
                if j > 0:
                    inp = self.embedding(inp) # M,embedding_dim
                hx, cx = self.lstm(inp, (hx, cx))
                # M,NUM_OPS -> seems to be M, NUM_OPS (M=8 ? where is M is
                # different instances of each input example augmented by
                # adverserial example)
                op = self.outop(hx)
                
                entropy, log_prob, action = self.calculate(op)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                
                inp = self.get_variable(action, requires_grad = False) # [M] ->[M]  
                inp = self.embedding(inp) # [M] ->[M, embedding_dim]
                hx, cx = self.lstm(inp, (hx, cx))
                mag = self.outmag(hx) # [M,NM_MAGS]
    
                entropy, log_prob, action = self.calculate(mag)
                entropies.append(entropy)
                log_probs.append(log_prob)
                policies.append(action)
                
                # Why NUM_OPS + action ??- doesnt this excede max(NUM_OPS) which is passed to the next iteration
                inp = self.get_variable(NUM_OPS + action, requires_grad = False) 
        
        entropies = torch.stack(entropies, dim = -1) ## M,Q*4
        log_probs = torch.stack(log_probs, dim = -1) ## M,Q*4
        policies = torch.stack(policies, dim = -1) # [M,Q*4]. 20 discrete parameters to form a whole policy
        # All the entropy values returned are very similar as entropy means how much memory in bits to hold the information ??
        # policies - [M, 20] -> random indices e.g. [13,  0, 13,  7,  4,  2, 13,  0, 11,  9,  2,  9,  3,  5,  1,  3,  2,  2, 10,  5]
        return policies, torch.sum(log_probs, dim = -1), torch.sum(entropies, dim = -1) # (M,Q*4) (M,) (M,) 
