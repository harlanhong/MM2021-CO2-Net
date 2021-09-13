import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import utils
import numpy as np
from torch.autograd import Variable
import time
import pdb
from tensorboard_logger import Logger
import ramps

from losses import *
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import losses


def train(itr, dataset, args, model, optimizer, logger, device):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    #interative
    pseudo_label = None
   
    outputs = model(features,seq_len=seq_len,is_training=True,itr=itr,opt=args)
    total_loss = model.criterion(outputs,labels,seq_len=seq_len,device=device,logger=logger,opt=args,itr=itr,pairs_id=pairs_id,inputs=features)
    # print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    #-----------
    # element_atn = outputs['attn']
    # bottom_attn_value, bottom_attn_index = torch.topk(element_atn, int(element_atn.shape[-2] // args.k//2), dim=1,largest=False)
    # top_attn_value, top_attn_index = torch.topk(element_atn, int(element_atn.shape[-2] // args.k//2), dim=1)
    # # attn_pseudo_loss = torch.abs(top_attn_value.mean()-1)+bottom_attn_value.mean()
    # attn_pseudo_loss = element_atn.mean(1).detach()*torch.abs(top_attn_value.mean(1)-1)+(1-element_atn.mean(1)).detach()*bottom_attn_value.mean(1)
    # weights = ramps.linear_rampup(itr,1000)
    # total_loss=total_loss+weights*attn_pseudo_loss.mean()
    #----------------------
    optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    return total_loss.data.cpu().numpy()

