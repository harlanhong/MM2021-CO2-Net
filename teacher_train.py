import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
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


def train(itr, dataset, args, model, optimizer, logger, device,teacher_model):
    model.train()
    features, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    
    features = torch.from_numpy(features).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    #interative
    pseudo_label = None

    outputs = model(features,seq_len=seq_len,is_training=True,itr=itr,opt=args)
    with torch.no_grad():
        teacher_outputs = teacher_model(features,seq_len=seq_len,is_training=True,itr=itr,opt=args)
    
    total_loss = model.criterion(outputs,labels,seq_len=seq_len,device=device,logger=logger,opt=args,itr=itr,pairs_id=pairs_id,inputs=features)
    pseudo_cas = teacher_outputs['cas']
    cas = outputs['cas']
    attn = outputs['attn']
    
    top_pred, top_index = torch.topk(attn, int(attn.shape[-2] // args.k), dim=1)
    n,b,_ = top_index.shape
    expand_top_index = top_index.expand(n,b,cas.shape[-1])
    select_top_value = torch.gather(cas,1,expand_top_index)
    teacher_top_value = torch.gather(pseudo_cas,1,expand_top_index)
    
    _, bottom_index = torch.topk(attn, int(attn.shape[-2] // args.k), dim=1,largest=False)
    n,b,_ = bottom_index.shape
    expand_bottom_index = top_index.expand(n,b,cas.shape[-1])
    select_bottom_value = torch.gather(cas,1,expand_top_index)
    teacher_bottom_value = torch.gather(pseudo_cas,1,expand_top_index)


    mseloss = nn.MSELoss(reduce=False)
    labels_with_back = torch.cat(
            (labels, torch.ones_like(labels[:, [0]])), dim=-1)
    
    b,n,c = cas.shape
    top_labels_with_back = labels_with_back.unsqueeze(1).expand_as(select_top_value)
    bottom_labels_with_back = labels_with_back.unsqueeze(1).expand_as(select_bottom_value)

    tmp =mseloss(select_bottom_value, teacher_bottom_value)*bottom_labels_with_back
    bottom_ema_loss = tmp.sum(-1)/bottom_labels_with_back.sum(-1)

    tmp =mseloss(select_top_value, teacher_top_value)*top_labels_with_back
    top_ema_loss = tmp.sum(-1)/top_labels_with_back.sum(-1)

    weights = ramps.linear_rampup(itr,1000)
    total_loss=total_loss+ (weights*(bottom_ema_loss.mean()+top_ema_loss.mean()))
    # print('Iteration: %d, Loss: %.3f' %(itr, total_loss.data.cpu().numpy()))
    optimizer.zero_grad()
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    return total_loss.data.cpu().numpy()

