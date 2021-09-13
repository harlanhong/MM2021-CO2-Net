
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_logger import log_value
import numpy as np
from torch.autograd import Variable
import time
import pdb
import math

torch.set_default_tensor_type('torch.cuda.FloatTensor')





def Contrastive_RandomSample(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
    element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
    seq_len should be numpy array of dimension (B,)
    labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''
    sim_loss = 0.
    n_tmp = 0.
    _, n, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i], dim=0)
        atn2 = F.softmax(element_logits[i+1], dim=0)

        n1 = torch.FloatTensor([np.maximum(n-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(n-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)      # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        # d1 = torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        # d2 = torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        # d3 = torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d2-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d3-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss

def Contrastive(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    _, _, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)      # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        # d1 = torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        # d2 = torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        # d3 = torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d2-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d3-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss


def Contrastive_Sigmoid(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    _, _, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        atn1 = F.sigmoid(element_logits[i][:seq_len[i]])
        neg_atn1=(1-atn1)/torch.sum(1-atn1,dim=0,keepdim=True)
        atn1=atn1/torch.sum(atn1,dim=0,keepdim=True)
        atn2 = F.sigmoid(element_logits[i+1][:seq_len[i+1]])
        neg_atn2=(1-atn2)/torch.sum(1-atn2,dim=0,keepdim=True)
        atn2=atn2/torch.sum(atn2,dim=0,keepdim=True)


        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)      # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), neg_atn1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), neg_atn2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        # d1 = torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        # d2 = torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        # d3 = torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d2-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d3-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss


def Contrastive_Batch(x, element_logits, seq_len, n_similar, labels, device,args):
    sim_loss = 0.
    n_tmp = 0.
    _, _, c = element_logits.shape

    positives=[]
    negatives=[]

    for i in range(n_similar*2):
        atn=torch.softmax(element_logits[i][:seq_len[i]],dim=0)
        n = torch.FloatTensor([np.maximum(seq_len[i] - 1, 1)]).to(device)
        Hf = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn)  # (n_feature, n_class)
        Lf = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn) / n)

        positives.append(Hf)
        negatives.append(Lf)

    positives=torch.stack(positives,dim=0)
    negatives=torch.stack(negatives,dim=0)
    # [B,C]
    # to calculate the distance matrix between of each videos
    PN_Dissim_Mat=1-torch.sum(positives.unsqueeze(1)*negatives.unsqueeze(0),dim=2)/(torch.norm(positives.unsqueeze(1),p=2,dim=2)*torch.norm(negatives.unsqueeze(0),p=2,dim=2))
    PP_Dissim_Mat=1-torch.sum(positives.unsqueeze(1)*positives.unsqueeze(0),dim=2)/(torch.norm(positives.unsqueeze(1),p=2,dim=2)*torch.norm(positives.unsqueeze(0),p=2,dim=2))

    for i in range(0,n_similar*2,2):
        neg_dis_vec_1=PN_Dissim_Mat[i]
        neg_dis_vec_2=PN_Dissim_Mat[i+1]
        pos_dis_vec_1=PP_Dissim_Mat[i]
        pos_dis_vec_2=PP_Dissim_Mat[i+1]
        pos_dis=pos_dis_vec_1[i]

        sim_loss=sim_loss + 0.25 * torch.sum(torch.relu(torch.max((-pos_dis_vec_1+pos_dis.unsqueeze(0)+0.5)* Variable(labels[i, :]) * Variable(
                labels[i + 1, :]),dim=0)[0]))
        sim_loss=sim_loss + 0.25 * torch.sum(torch.relu(torch.max((-pos_dis_vec_2+pos_dis.unsqueeze(0)+0.5)* Variable(labels[i, :]) * Variable(
                labels[i + 1, :]),dim=0)[0]))
        # Hf1=positives[i]
        # Hf2=positives[i+1]
        #
        # pos_dis=1-torch.sum(Hf1*Hf2,dim=0)/(torch.norm(Hf1,2,dim=0)*torch.norm(Hf2,2,dim=0))

        sim_loss = sim_loss + 0.25 * torch.sum(torch.relu(torch.max((-neg_dis_vec_1+pos_dis.unsqueeze(0)+0.5)* Variable(labels[i, :]) * Variable(
                labels[i + 1, :]),dim=0)[0]))
        sim_loss = sim_loss + 0.25 * torch.sum(torch.relu(torch.max((-neg_dis_vec_2+pos_dis.unsqueeze(0)+0.5)* Variable(labels[i, :]) * Variable(
                labels[i + 1, :]),dim=0)[0]))

        # sim_loss=sim_loss+0.5*torch.sum(torch.relu(pos_dis-torch.max(neg_dis_vec_1,dim=0)[0]+0.5) * Variable(labels[i, :]) * Variable(
        #         labels[i + 1, :]))
        # sim_loss=sim_loss+0.5*torch.sum(torch.relu(pos_dis-torch.max(neg_dis_vec_2,dim=0)[0]+0.5)* Variable(labels[i, :]) * Variable(
        #         labels[i + 1, :]))

        n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])

    sim_loss = sim_loss / n_tmp
    return sim_loss

def Contrastive_HT(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    '''Hard tanh is used in computing the attention'''

    sim_loss = 0.
    n_tmp = 0.
    _, _, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(F.hardtanh(element_logits[i][:seq_len[i]],-args.clip,args.clip), dim=0)
        atn2 = F.softmax(F.hardtanh(element_logits[i+1][:seq_len[i+1]],-args.clip,args.clip), dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)

        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)      # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        # d1 = torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        # d2 = torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        # d3 = torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d2-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d3-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss

def Contrastive_TopK(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    '''Hard tanh is used in computing the attention'''

    sim_loss = 0.
    n_tmp = 0.
    _, _, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        # atn1 = F.softmax(F.hardtanh(element_logits[i][:seq_len[i]],-args.clip,args.clip), dim=0)
        # atn2 = F.softmax(F.hardtanh(element_logits[i+1][:seq_len[i+1]],-args.clip,args.clip), dim=0)

        # select topk to construct a attention
        atn1=torch.sigmoid(element_logits[i][:seq_len[i]])
        atn2=torch.sigmoid(element_logits[i+1][:seq_len[i+1]])

        

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)

        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)      # (n_feature, n_class)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        # d1 = torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        # d2 = torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        # d3 = torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d2-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        # sim_loss = sim_loss + 0.5*torch.sum(torch.max(d3-d1+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss



def Inner_Outer_Loss(element_logits,seq_len,batch_size,labels,device,args,threshold=0.5,):
    # use as a regularization in training, inner similar, outer dissimilar
    io_loss=0
    for i in range(batch_size):
        # [N,C]
        scores=torch.softmax(element_logits[i][:seq_len[i]],dim=-1)
        # to select the gt_class (inner and outer should be clear different
        label=labels[i]
        # [C]->[]
        label_mask=label.unsqueeze(-1).repeat(1,seq_len[i]).bool().permute([1,0])
        selected_scores=torch.masked_select(scores,label_mask)
        vid_pred=[]
        for scores in selected_scores:
            # [N,X]
            thre=torch.max(scores)[0]-(torch.max(scores)[0]-torch.min(scores)[0])*threshold
            # to generate segments
            vid_pred=torch.cat([np.zeros])
        selected_scores=torch.ge(selected_scores,threshold)


def rankingloss1(element_logits, seq_len, batch_size, labels, device,args,pair_id):
    b,n,c = element_logits.shape
    element_prob = F.softmax(element_logits,-1)
    # bg_prob = element_prob[:,:,-1].unsqueeze(-1)
    loss = 0
    k = np.ceil(seq_len/8).astype('int32')

    for i in range(args.num_similar):
        Pos = [torch.topk(element_logits[i*args.similar_size+k][:seq_len[i*args.similar_size+k],pair_id[i]],k=int(k[i*args.similar_size+k]))[0] for k in range(args.similar_size)]
        Neg = [torch.max(element_logits[(i+args.num_similar)*args.similar_size+k][:seq_len[(i+args.num_similar)*args.similar_size+k],pair_id[i]]) for k in range(args.similar_size)]
        pdb.set_trace()

        min_pos = torch.min(Pos)
        max_neg = torch.max(Neg)
        s_loss = F.relu(max_neg-min_pos+1,0)
        loss += s_loss
    loss/=args.num_similar
    return loss

def rankingloss(element_logits, seq_len, batch_size, labels, device,args,pair_id):
    b,n,c = element_logits.shape
    element_prob = F.softmax(element_logits,-1)
    # bg_prob = element_prob[:,:,-1].unsqueeze(-1)
    loss = 0
    for i in range(args.num_similar):
        Pos = torch.stack([torch.max(element_prob[i*args.similar_size+k][:seq_len[i*args.similar_size+k],pair_id[i]]) for k in range(args.similar_size)])
        Neg = torch.stack([torch.max(element_prob[(i+args.num_similar)*args.similar_size+k][:seq_len[(i+args.num_similar)*args.similar_size+k],pair_id[i]]) for k in range(args.similar_size)])

        min_pos = torch.min(Pos)
        max_neg = torch.max(Neg)
        s_loss = F.relu(max_neg-min_pos+0.5,0)
        loss += s_loss
    loss/=args.num_similar
    return loss

def bg_contraint(element_logits, seq_len, batch_size, labels, device,args):
    b,n,c = element_logits.shape
    element_prob = F.softmax(element_logits,-1)
    bg_prob = element_prob[:,:,-1].unsqueeze(-1)
    loss = 0
    for i in range(b):
        bgs = bg_prob[i,:seq_len[i]].sum()
        gt_nums = torch.tensor(args.bg_prob*seq_len[i],requires_grad=False).float()
        loss += (torch.pow(bgs-gt_nums,2)/seq_len[i])
    loss/=b
    return loss
    # entropy =

def self_bg_supervised(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    element_prob = F.softmax(element_logits,-1)
    act_prob = F.softmax(element_logits[:,:,:-1],-1)
    entropy = -torch.sum(act_prob*torch.log(act_prob),dim=-1,keepdim=True)
    Max_entropy = -np.log(1/20)
    bg_weight = 1-entropy/Max_entropy
    bg_prob = element_prob[:,:,-1].unsqueeze(-1)
    loss = 0
    for i in range(b):
        loss+=torch.mean(torch.pow((bg_prob[:seq_len[i]]-bg_weight[:seq_len[i]]),2))
    loss/=b
    return loss
    # entropy =

def NegLoss2(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    nagetive_loss = 0
    for i in range(b):
        s_logits = element_logits[i][:seq_len[i]]
        s_label = labels[i]
        neg_label = (s_label==0).float()
        s_prob = F.sigmoid(s_logits)
        # selected_prob = s_prob[:,neg_idx]
        inverse_prob = 1- s_prob
        s_loss = -(neg_label*torch.log(inverse_prob)).sum(1).mean()
        nagetive_loss+=s_loss
    nagetive_loss/=b
    return nagetive_loss

def NegLoss1(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    nagetive_loss = 0
    for i in range(b):
        s_logits = element_logits[i][:seq_len[i]]
        s_label = labels[i]
        pdb.set_trace()
        neg_label = 1-s_label
        s_prob = F.softmax(s_logits,-1)
        inverse_prob = 1- s_prob
        s_loss = -(neg_label*torch.log(inverse_prob)).sum(1).mean()
        nagetive_loss+=s_loss
    nagetive_loss/=b
    return nagetive_loss

def NegLoss_RandomSample(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    negative_loss = 0
    for i in range(b):
        s_logits = element_logits[i]
        s_label = labels[i]
        neg_idx = s_label==0
        s_prob = F.sigmoid(s_logits)
        selected_prob = s_prob[:,neg_idx]
        inverse_prob = 1 - selected_prob
        s_loss = -torch.log(inverse_prob+1e-8).sum(1).mean()
        negative_loss+=s_loss
    negative_loss/=b
    return negative_loss

def NegLoss(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    negative_loss = 0
    for i in range(b):
        s_logits = element_logits[i][:seq_len[i]]
        s_label = labels[i]
        neg_idx = s_label==0
        s_prob = F.sigmoid(s_logits)
        selected_prob = s_prob[:,neg_idx]
        inverse_prob = 1 - selected_prob
        s_loss = -torch.log(inverse_prob+1e-8).sum(1).mean()
        negative_loss+=s_loss
    negative_loss/=b
    return negative_loss

# =========================== Conv ATT Loss ================================

def NegLoss_Vid(element_logits,seq_len,batch_size,labels,device,args):
    negloss=0
    element_logits=element_logits[0]
    b=element_logits.shape[0]

    #[B,C]
    for i in range(b):
        s_label = labels[i]
        neg_idx = s_label == 0
        s_prob=torch.softmax(element_logits[i],dim=-1)
        selected_prob = s_prob[neg_idx]
        inverse_prob = 1 - selected_prob
        s_loss = -torch.log(inverse_prob).sum()
        negloss+=s_loss
    negloss/=b
    return negloss

def NegLoss_Clip(element_logits,seq_len,batch_size,labels,device,args):
    element_logits=element_logits[-1]
    return NegLoss(element_logits,seq_len,batch_size,labels,device)

def SpaLoss(element_logits,seq_len,batch_size,labels,devices,args):
    element_logits=element_logits[1]
    return torch.mean(element_logits)

def GapLoss(element_logits,seq_len,batch_size,labels,devices,args):
    gaploss=0
    element_logits=element_logits[1]

    for i in range(seq_len.shape[0]):
        att=element_logits[i][:,:seq_len[i]].squeeze(0)
        l=max(1,int(seq_len[i]/args.s))
        # selected_att=np.random.choice(att,l)
        att=torch.sort(att,dim=0)[0]
        gaploss+=torch.relu(att[:l].mean()-att[-l:].mean()+1)
    gaploss=gaploss/seq_len.shape[0]
    return gaploss

def MILL_Vid(element_logits,seq_len,batch_size,labels,device,args):
    element_logits=element_logits[0]
    # labels is with shape [B,C]
    millloss=-torch.mean(torch.sum(labels*F.log_softmax(element_logits,dim=1),dim=1),dim=0)
    return millloss

def MILL_Clip(element_logits,seq_len,batch_size,labels,device,args):
    element_logits=element_logits[-1]
    return MILL(element_logits,seq_len,batch_size,labels,device,args)

def Contractive_Att(x,att_scores,seq_len,n_similar,labels,device,args):
    sim_loss=0
    for i in range(0,n_similar*2,2):
        # []
        Hf1 = torch.sum((x[i]*att_scores[i])[:seq_len[i]],dim=0)/torch.sum(att_scores[i][:seq_len[i]])
        Hf2 = torch.sum((x[i+1]*att_scores[i+1])[:seq_len[i+1]],dim=0)/torch.sum(att_scores[i+1][:seq_len[i+1]])

        Lf1 = torch.sum((x[i]*(1-att_scores[i]))[:seq_len[i]],dim=0)/torch.sum((1-att_scores[i])[:seq_len[i]])
        Lf2 = torch.sum((x[i+1]*(1-att_scores[i+1]))[:seq_len[i+1]],dim=0)/torch.sum((1-att_scores[i+1])[:seq_len[i+1]])

        d1=torch.sum(Hf1*Hf2)/(torch.norm(Hf1,p=2,dim=0)*torch.norm(Hf2,p=2,dim=0))
        d2=torch.sum(Hf1*Lf1)/(torch.norm(Hf1,p=2,dim=0)*torch.norm(Lf1,p=2,dim=0))
        d3=torch.sum(Hf1*Lf2)/(torch.norm(Hf1,p=2,dim=0)*torch.norm(Lf2,p=2,dim=0))
        sim_loss+=0.5*torch.relu(d1-d2+0.5)
        sim_loss+=0.5*torch.relu(d1-d3+0.5)

    return sim_loss/n_similar

def Contractive_Vid(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    _,fa,fb=x
    # _,att_logits=element_logits
    # _, _, c = element_logits.shape
    for i in range(0, n_similar*2, 2):
        # [1,C]
        Hf1=fa[i]
        Hf2=fa[i+1]
        Lf1=fb[i]
        Lf2=fb[i+1]

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))        # 1-similarity
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.relu(d1-d2+0.5)*torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))/(torch.sum(labels[i,:])*torch.sum(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.relu(d1-d3+0.5)*torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))/(torch.sum(labels[i,:])*torch.sum(labels[i+1,:]))

        n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
    sim_loss = sim_loss / n_tmp
    return sim_loss

def Contractive_Clip(x,element_logits, seq_len, n_similar, labels, device,args):
    feat_embed,_,_=x
    _,_,clip_cls_logits=element_logits
    return Contrastive(feat_embed,clip_cls_logits,seq_len,n_similar,labels,device,args)

# =========================================================================
# for asymmetric fusion
def MSE_Loss(a_logits,b_logits,seq_len,batch_size,labels,device,args):
    mse_loss=0

    for i,sl in enumerate(seq_len):
        mse_loss+=torch.mean((a_logits[i]-b_logits[i])**2[:,:sl])
    return mse_loss/seq_len.shape[0]

# ========================================================================
def MILL2(element_logits, seq_len, batch_size, labels, device, args):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''
    k = np.ceil(seq_len/args.k).astype('int32')
    b, n_class = labels.shape
    # labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    element_prob = F.sigmoid(element_logits)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_prob[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = torch.norm(instance_logits-labels)
    # milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def MILL1(element_logits, seq_len, batch_size, labels, device,args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    instance_logits = torch.zeros(0).to(device)
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(element_logits.shape[0]):
        peaks = n_peak_find(
            element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        )
        instance_logits = torch.cat([instance_logits, torch.mean(peaks, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def MILL(element_logits, seq_len, batch_size, labels, device, args):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/args.k).astype('int32')
    b, n_class = labels.shape
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

# not work well
def MILL_Con(element_logits, seq_len, batch_size, labels, device, args):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/args.k).astype('int32')
    b, n_class = labels.shape
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        # import pdb
        # pdb.set_trace()
        ele_logits=element_logits[i][:seq_len[i]]
        ele_logits=torch.cat([ele_logits[0].unsqueeze(0).repeat([3,1]),ele_logits,ele_logits[-1].unsqueeze(0).repeat([2,1])],dim=0)
        smo_ele_logits=F.avg_pool1d(ele_logits.unsqueeze(0).permute(0,2,1),kernel_size=6,stride=1)[0].permute([1,0])
        tmp, _ = torch.topk(smo_ele_logits, k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def MYMILL(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''
    element_prob = F.softmax(element_logits,-1)
    bg_prob = element_prob[:,:,-1].unsqueeze(2)
    element_prob = element_prob[:,:,:-1]
    element_prob = element_prob*(element_prob>bg_prob).long()
    element_prob = torch.cat((element_prob,bg_prob),2)
    element_logits = element_logits*element_prob
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp = element_logits[i][:seq_len[i]]
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def RESHAPE_MILL(element_logits, seq_len, batch_size, labels, device):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''
    k = np.ceil(seq_len/8).astype('int32')
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def EMILL(element_logits, seq_len, batch_size, labels, device):
    k = np.ceil(seq_len/8).astype('int32')
    # pdb.set_trace()
    b,c = labels.shape
    ones = torch.ones(b,1).to(device)
    labels = torch.cat((labels,ones),1).detach()
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(batch_size):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(Variable(labels) * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def ECASL(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''
    b,c = labels.shape
    sim_loss = 0.
    n_tmp = 0.
    element_prob = F.softmax(element_logits,-1)
    prob_ac = element_prob[:,:,:-1]
    prob_bg = element_prob[:,:,-1].unsqueeze(2)
    mask = prob_ac>prob_bg
    refine_prob_ac = prob_ac*mask.long()
    contribute_ac = refine_prob_ac*(1-prob_bg)
    labels = labels[:,:-1]
    pdb.set_trace()
    for i in range(0, n_similar*2, 2):
        atn1 = contribute_ac[i][:seq_len[i]]
        atn2 = contribute_ac[i+1][:seq_len[i+1]]

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))
    sim_loss = sim_loss / n_tmp
    return sim_loss

def CASL(x, element_logits, seq_len, n_similar, labels, device,args):
    ''' x is the torch tensor of feature from the last layer of model of dimension (n_similar, n_element, n_feature),
        element_logits should be torch tensor of dimension (n_similar, n_element, n_class)
        seq_len should be numpy array of dimension (B,)
        labels should be a numpy array of dimension (B, n_class) of 1 or 0 '''

    sim_loss = 0.
    n_tmp = 0.
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]], dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)

        n1 = torch.FloatTensor([np.maximum(seq_len[i]-1, 1)]).to(device)
        n2 = torch.FloatTensor([np.maximum(seq_len[i+1]-1, 1)]).to(device)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2)
        Lf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), (1 - atn1)/n1)
        Lf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), (1 - atn2)/n2)

        d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
        d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
        d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).to(device))*Variable(labels[i,:])*Variable(labels[i+1,:]))
        n_tmp = n_tmp + torch.sum(Variable(labels[i,:])*Variable(labels[i+1,:]))
    sim_loss = sim_loss / n_tmp
    return sim_loss

def n_peak_find(x, args, win_size):
    # x : L, C
    split_list = [win_size] * (x.shape[0] // win_size - 1) + [
        x.shape[0] - max(0, win_size * (x.shape[0] // win_size - 1))
    ]
    splits = torch.split(x, split_list, dim=0)
    peak = torch.Tensor()
    for x_split in splits:
        sp_topk = torch.topk(x_split, int(min(x_split.shape[0], args.topk2)), dim=0)[
            0
        ]
        mean_sp = torch.mean(sp_topk, 0, keepdim=True)
        peak = torch.cat((peak, mean_sp))
    return peak

def mill_loss(element_logits, seq_len, batch_size,labels, device, args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)

    for i in range(element_logits.shape[0]):
        peaks = n_peak_find(
            element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        )

        block_peak = torch.sigmoid(peaks)
        prob = 1 - torch.prod(1 - block_peak, 0)
        lab = labels[i]
        loss1 = -torch.sum(lab * torch.log(prob + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
            1 - lab
        )
        loss += 1 / 2 * (loss1 + loss2)
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss

def pos_only_mill_loss(element_logits,seq_len,batch_size,labels,device,args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)

    for i in range(element_logits.shape[0]):
        peaks = n_peak_find(
            element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        )

        block_peak = torch.sigmoid(peaks)
        prob = 1 - torch.prod(1 - block_peak, 0)
        # prob 1 -> one of them is 1
        # prob 0 -> none of them is 1
        lab = labels[i]
        # label as mask to select the gt class for latter training
        # tar_porb=torch.masked_select(prob,lab.unsuqeeze(-1)).view(torch.sum(lab),-1)
        # loss+=-torch.sum(torch.log(tar_prob+eps)/torch.sum(lab))
        loss1 = -torch.sum(lab * torch.log(prob + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
            1 - lab
        )
        loss += 1 / 2 * (loss1 + loss2)
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss

def focal_mill_loss(element_logits,seq_len,batch_size,labels,device,args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)

    for i in range(element_logits.shape[0]):
        peaks = n_peak_find(
            element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        )

        block_peak = torch.sigmoid(peaks)
        prob = 1 - torch.prod(1 - block_peak, 0)
        lab = labels[i]

        loss1=-torch.sum(args.alpha*(1-prob)**args.gamma*lab*torch.log(prob+eps)/torch.sum(lab))
        loss2=-torch.sum((1-args.alpha)*(prob)**args.gamma*(1-lab)*torch.log(1-prob+eps)/torch.sum(1-lab))

        # loss1 = -torch.sum((1-prob)**args.gamma*lab * torch.log(prob + eps)) / torch.sum(lab)
        # loss2 = -torch.sum((prob**args.gamma*(1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
        #     1 - lab
        # )
        # loss += 1 / 2 * (loss1 + loss2)
        loss=loss1+loss2
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss

def attn_n_peak_find(x,attn,args,win_size):
    split_list = [win_size] * (x.shape[0] // win_size - 1) + [
        x.shape[0] - max(0, win_size * (x.shape[0] // win_size - 1))
    ]
    x_splits=torch.split(x,split_list,dim=0)
    attn_splits=torch.split(attn,split_list,dim=0)
    # sigmoid the x
    peak = torch.Tensor()
    for x_split,attn_split in zip(x_splits,attn_splits):
        # sp_topk,topk_index=torch.topk(x_split,int(min(x_split.shape[0],args.topk2)),dim=0)
        # attn_topk=attn[topk_index]
        attn_topk,topk_index=torch.topk(attn_split,int(min(attn_split.shape[0],args.topk2)),dim=0)
        sp_topk=x_split[topk_index]
        mean_sp=torch.sigmoid(torch.mean(sp_topk,0,keepdim=True))*torch.sigmoid(torch.mean(attn_topk))
        # [10,20] [10,20,1]
        # mean_sp=torch.mean(torch.sigmoid(sp_topk)*torch.sigmoid(attn_topk).squeeze(2),0,keepdim=True)
        peak = torch.cat((peak, mean_sp))
    return peak

def focal_mill_attn_loss(element_logits,attn,seq_len,batch_size,labels,device,args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)

    for i in range(element_logits.shape[0]):
        peaks = attn_n_peak_find(
            element_logits[i][: seq_len[i]],attn[i][:seq_len[i]], args, win_size=int(args.topk)
        )
        # block_peak = torch.sigmoid(peaks)
        block_peak=peaks
        prob = 1 - torch.prod(1 - block_peak, 0)
        lab = labels[i]

        loss1=-torch.sum(args.alpha*(1-prob)**args.gamma*lab*torch.log(prob+eps)/torch.sum(lab))
        loss2=-torch.sum((1-args.alpha)*(prob)**args.gamma*(1-lab)*torch.log(1-prob+eps)/torch.sum(1-lab))

        # loss1 = -torch.sum((1-prob)**args.gamma*lab * torch.log(prob + eps)) / torch.sum(lab)
        # loss2 = -torch.sum((prob**args.gamma*(1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
        #     1 - lab
        # )
        # loss += 1 / 2 * (loss1 + loss2)
        loss=loss1+loss2
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss

def focal_mill_attn_loss_RandomSample(element_logits,attn,seq_len,batch_size,labels,device,args):
    # k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)

    for i in range(element_logits.shape[0]):
        peaks = attn_n_peak_find(
            element_logits[i],attn[i], args, win_size=int(args.topk)
        )
        # block_peak = torch.sigmoid(peaks)
        block_peak=peaks
        prob = 1 - torch.prod(1 - block_peak, 0)
        lab = labels[i]

        loss1=-torch.sum(args.alpha*(1-prob)**args.gamma*lab*torch.log(prob+eps)/torch.sum(lab))
        loss2=-torch.sum((1-args.alpha)*(prob)**args.gamma*(1-lab)*torch.log(1-prob+eps)/torch.sum(1-lab))

        # loss1 = -torch.sum((1-prob)**args.gamma*lab * torch.log(prob + eps)) / torch.sum(lab)
        # loss2 = -torch.sum((prob**args.gamma*(1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
        #     1 - lab
        # )
        # loss += 1 / 2 * (loss1 + loss2)
        loss=loss1+loss2
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss

def mill_loss2(element_logits, seq_len, batch_size,labels, device,args):
    k = np.ceil(seq_len / args.topk).astype("int32")
    eps = 1e-8
    loss = 0
    # element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    k = np.ceil(seq_len/8).astype('int32')

    for i in range(element_logits.shape[0]):
        # peaks = n_peak_find(
            # element_logits[i][: seq_len[i]], args, win_size=int(args.topk)
        # )
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        peaks = torch.mean(tmp, 0, keepdim=True)
        block_peak = F.softmax(peaks,1)

        prob = 1 - torch.prod(1 - block_peak, 0)

        lab = labels[i]
        loss1 = -torch.sum(lab * torch.log(prob + eps)) / torch.sum(lab)
        loss2 = -torch.sum((1 - lab) * torch.log(1 - prob + eps)) / torch.sum(
            1 - lab
        )
        loss += 1 / 2 * (loss1 + loss2)
        # loss +=loss1
    milloss = loss / element_logits.shape[0]
    return milloss
    return milloss

def mill_loss3(element_logits, seq_len, labels, device,args):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        tmp, _ = torch.topk(element_logits[i][:seq_len[i]], k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def get_unit_vector(x, dim=0):
    return x / torch.norm(x, 2, dim=dim, keepdim=True)


def batch_per_dis(X1, X2, w):
    X1 = X1.permute(2, 1, 0).unsqueeze(2)
    X2 = X2.permute(2, 1, 0).unsqueeze(1)

    X_d = X1 - X2
    X_diff = X_d.reshape(X_d.shape[0], X_d.shape[1] * X_d.shape[2], -1) #差值

    w = w.unsqueeze(-1)
    dis_mat = torch.bmm(X_diff, w).squeeze(-1)  #attention的东西
    dis_mat = dis_mat.reshape(dis_mat.shape[0], X_d.shape[1], X_d.shape[2])
    dis_mat = torch.pow(dis_mat, 2) #二次
    return dis_mat

def metric_loss(x, element_logits, weight, labels, seq_len, device, args):

    sim_loss = 0.0
    labels = labels
    n_tmp = 0.0
    # element_logits = element_logits[:,:,:-1]
    # labels = labels[:,:-1]
    element_logits = F.hardtanh(element_logits, -args.clip, args.clip)
    for i in range(0, args.num_similar * args.similar_size, args.similar_size):
        lab = labels[i, :]
        #lab 包含共同出现的action
        for k in range(i + 1, i + args.similar_size):
            lab = lab * labels[k, :]
        common_ind = lab.nonzero().squeeze(-1)
        Xh = torch.Tensor()
        Xl = torch.Tensor()

        for k in range(i, i + args.similar_size):
            elem = element_logits[k][: seq_len[k], common_ind]
            atn = F.softmax(elem, dim=0)  #
            n1 = torch.FloatTensor([np.maximum(seq_len[k] - 1, 1)]).to(device)
            atn_l = (1 - atn) / n1
            xh = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn) #算动作A的加权平均特征  公式10
            xl = torch.mm(torch.transpose(x[k][: seq_len[k]], 1, 0), atn_l) ##非动作A的加权平均特征
            xh = xh.unsqueeze(1)
            xl = xl.unsqueeze(1)
            Xh = torch.cat([Xh, xh], dim=1)
            Xl = torch.cat([Xl, xl], dim=1)
        Xh = get_unit_vector(Xh, dim=0)
        Xl = get_unit_vector(Xl, dim=0)

        D1 = batch_per_dis(Xh, Xh, weight[common_ind, :])

        D1 = torch.triu(D1, diagonal=1)
        D1 = D1.view(D1.shape[0], -1)
        d1 = torch.sum(D1, -1) / (
            args.similar_size * (args.similar_size - 1) / 2
        )

        D2 = batch_per_dis(Xh, Xl, weight[common_ind, :])

        D2 = D2 * (1 - torch.eye(D2.shape[1])).unsqueeze(0)
        D2 = D2.view(D2.shape[0], -1)

        d2 = torch.sum(D2, -1) / (args.similar_size * (args.similar_size - 1))

        loss = torch.sum(
            torch.max(d1 - d2 + args.dis, torch.FloatTensor([0.0]).to(device))
        )
        sim_loss += loss
        n_tmp = n_tmp + torch.sum(lab)

    sim_loss = sim_loss / n_tmp
    return sim_loss


def smooth_loss1(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    total_loss = 0.0
    for i in range(b):
        tmp_logits = element_logits[i][: seq_len[i]]
        sq_loss = torch.pow(tmp_logits[:-1]-tmp_logits[1:],2)
        tmp_loss = torch.mean(torch.sum(sq_loss,0))
        total_loss+=tmp_loss
    return total_loss/b

def smooth_loss2(element_logits, seq_len, batch_size, labels, device):
    b,n,c = element_logits.shape
    new_element_logits = element_logits.permute(0,2,1).contiguous()
    avg_filter = F.avg_pool1d(new_element_logits,3,1,1)
    loss = torch.pow(new_element_logits-avg_filter,2)
    loss = torch.mean(loss)

    return loss

def norm_loss(x, element_logits, weight, labels, seq_len, device, args,pairs_id):
    element_prob = F.sigmoid(element_logits)

    return torch.mean(torch.abs(element_prob))


def ConT1(x, element_logits, seq_len, n_similar, labels, device,args,pairs_id):
    sim_loss = 0.
    n_tmp = 0.
    # bg = labels[:,-1]
    labels = labels[:,:-1]
    element_logits = element_logits[:,:,:-1]
    fake_labels = torch.zeros(n_similar*2,args.num_class)
    for i in range(n_similar):
        fake_labels[2*i][pairs_id[i]] = 1
        fake_labels[2*i+1][pairs_id[i]] = 1
    Feats = []
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]],dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1).permute(1,0)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2).permute(1,0)
        common_labels = (fake_labels[i,:]*fake_labels[i+1,:]).bool()
        Feats.append(Hf1[common_labels])
        Feats.append(Hf2[common_labels])

    Feats = torch.cat(Feats)
    feat_feat = torch.mm(Feats,Feats.permute(1,0))
    feat_norm = torch.norm(Feats, 2, dim=1)
    sim = feat_feat / (feat_norm.unsqueeze(0) * feat_norm.unsqueeze(1))
    sim = torch.exp(sim/args.tao)
    eye = 1-torch.eye(n_similar*2)
    entropy = -(sim*eye)/(sim*eye).sum(1).unsqueeze(1)
    total_loss = 0
    for i in range(n_similar):
        lij = entropy[2*i][2*i+1]+entropy[2*i+1][2*i]
        total_loss+=lij
    return total_loss/(n_similar*2)


def ConT2(x, element_logits, seq_len, n_similar, labels, device,args,pairs_id):
    sim_loss = 0.
    n_tmp = 0.
    bg_logit = element_logits[:,:,-1].unsqueeze(2)
    labels = labels[:,:-1]
    element_logits = element_logits[:,:,:-1]
    element_logits = element_logits*(element_logits>bg_logit).long()
    fake_labels = torch.zeros(n_similar*2,args.num_class)
    for i in range(n_similar):
        fake_labels[2*i][pairs_id[i]] = 1
        fake_labels[2*i+1][pairs_id[i]] = 1
    Feats = []
    for i in range(0, n_similar*2, 2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]],dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1).permute(1,0)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2).permute(1,0)
        common_labels = (fake_labels[i,:]*fake_labels[i+1,:]).bool()
        Feats.append(Hf1[common_labels])
        Feats.append(Hf2[common_labels])

    Feats = torch.cat(Feats)
    feat_feat = torch.mm(Feats,Feats.permute(1,0))
    feat_norm = torch.norm(Feats, 2, dim=1)
    sim = feat_feat / (feat_norm.unsqueeze(0) * feat_norm.unsqueeze(1))
    sim = torch.exp(sim/args.tao)
    eye = 1-torch.eye(n_similar*2)
    entropy = -(sim*eye)/(sim*eye).sum(1).unsqueeze(1)
    total_loss = 0
    for i in range(n_similar):
        lij = entropy[2*i][2*i+1]+entropy[2*i+1][2*i]
        total_loss+=lij
    return total_loss/(n_similar*2)


def ConT3(x, element_logits, seq_len, n_similar, labels, device,args,pairs_id):
    sim_loss = 0.
    n_tmp = 0.
    # bg_logit = element_logits[:,:,-1].unsqueeze(2)
    labels = labels[:,:-1]
    element_logits = element_logits[:,:,:-1]
    # element_logits = element_logits*(element_logits>bg_logit).long()
    fake_labels = torch.zeros(n_similar*2,args.num_class)
    for i in range(n_similar):
        fake_labels[2*i][pairs_id[i]] = 1
        fake_labels[2*i+1][pairs_id[i]] = 1
    P1 = []
    P2 = []
    N1 = []
    N2 = []
    for i in range(0, n_similar*2,2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]],dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1).permute(1,0)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2).permute(1,0)

        atnN1 = F.softmax(element_logits[n_similar*2+i][:seq_len[n_similar*2+i]],dim=0)
        atnN2 = F.softmax(element_logits[n_similar*2+i+1][:seq_len[n_similar*2+i+1]], dim=0)
        HN1 = torch.mm(torch.transpose(x[n_similar*2+i][:seq_len[n_similar*2+i]], 1, 0), atnN1).permute(1,0)
        HN2 = torch.mm(torch.transpose(x[n_similar*2+i+1][:seq_len[n_similar*2+i+1]], 1, 0), atnN2).permute(1,0)

        common_labels = (fake_labels[i,:]*fake_labels[i+1,:]).bool()
        P1.append(Hf1[common_labels])
        P2.append(Hf2[common_labels])
        N1.append(HN1[common_labels])
        N2.append(HN2[common_labels])

    P1 = torch.cat(P1)
    P2 = torch.cat(P2)
    N1 = torch.cat(N1)
    N2 = torch.cat(N2)
    P1_norm = torch.norm(P1, 2, dim=1)
    P2_norm = torch.norm(P2, 2, dim=1)
    N1_norm = torch.norm(N1, 2, dim=1)
    N2_norm = torch.norm(N2, 2, dim=1)

    p1_n1 = torch.mm(P1,N1.permute(1,0))
    p1_p2 = torch.mm(P1,P2.permute(1,0))
    p1_n1_sim = 1-p1_n1 / (P1_norm.unsqueeze(1) * N1_norm.unsqueeze(0))
    p1_P2_sim = 1-p1_n1 / (P1_norm.unsqueeze(1) * P2_norm.unsqueeze(0))

    n1_p2 = torch.mm(N1,P2.permute(1,0))
    n1_n2 = torch.mm(N1,N2.permute(1,0))
    n1_p2_sim = 1-p1_n1 / (N1_norm.unsqueeze(1) * P2_norm.unsqueeze(0))
    n1_n2_sim = 1-p1_n1 / (N1_norm.unsqueeze(1) * N2_norm.unsqueeze(0))

    n2_p2 = torch.mm(N2,P2.permute(1,0))
    n2_n1 = torch.mm(N2,N1.permute(1,0))
    n2_p2_sim = 1-n2_p2 / (N2_norm.unsqueeze(1) * P2_norm.unsqueeze(0))
    n2_n1_sim = 1-n2_n1 / (N2_norm.unsqueeze(1) * N1_norm.unsqueeze(0))

    l1 = torch.mean(F.relu(p1_P2_sim-p1_n1_sim+0.5))
    l2 = torch.mean(F.relu(n1_n2_sim-n1_p2_sim+0.5))
    l3 = torch.mean(F.relu(n2_n1_sim-n2_p2_sim+0.5))

    return (l1+l2+l3)/3


def variance_metric_loss(x, element_logits, weight,seq_len, n_similar, labels, device,args,pairs_id):
    sim_loss = 0.
    n_tmp = 0.
    # bg_logit = element_logits[:,:,-1].unsqueeze(2)
    labels = labels[:,:-1]
    element_logits = element_logits[:,:,:-1]
    # element_logits = element_logits*(element_logits>bg_logit).long()
    fake_labels = torch.zeros(n_similar*2,args.num_class)
    for i in range(n_similar):
        fake_labels[2*i][pairs_id[i]] = 1
        fake_labels[2*i+1][pairs_id[i]] = 1
    P1 = []
    P2 = []
    N1 = []
    N2 = []
    for i in range(0, n_similar*2,2):
        atn1 = F.softmax(element_logits[i][:seq_len[i]],dim=0)
        atn2 = F.softmax(element_logits[i+1][:seq_len[i+1]], dim=0)
        Hf1 = torch.mm(torch.transpose(x[i][:seq_len[i]], 1, 0), atn1).permute(1,0)
        Hf2 = torch.mm(torch.transpose(x[i+1][:seq_len[i+1]], 1, 0), atn2).permute(1,0)

        atnN1 = F.softmax(element_logits[n_similar*2+i][:seq_len[n_similar*2+i]],dim=0)
        atnN2 = F.softmax(element_logits[n_similar*2+i+1][:seq_len[n_similar*2+i+1]], dim=0)
        HN1 = torch.mm(torch.transpose(x[n_similar*2+i][:seq_len[n_similar*2+i]], 1, 0), atnN1).permute(1,0)
        HN2 = torch.mm(torch.transpose(x[n_similar*2+i+1][:seq_len[n_similar*2+i+1]], 1, 0), atnN2).permute(1,0)

        common_labels = (fake_labels[i,:]*fake_labels[i+1,:]).bool()
        P1.append(Hf1[common_labels])
        P2.append(Hf2[common_labels])
        N1.append(HN1[common_labels])
        N2.append(HN2[common_labels])

    P1 = torch.cat(P1)
    P2 = torch.cat(P2)
    N1 = torch.cat(N1)
    N2 = torch.cat(N2)


    PosP = P1-P2
    PosN = N1-N2
    NegPn1 = P1-N1
    NegPn2 = P1-N2
    NegNp1 = N1-P1
    NegNp1 = N1-P2
    # pdb.set_trace()
    w = weight[pairs_id,:]
    pos_diff_1 = (PosP*w).sum(1)
    neg_diff_1 = (NegPn1*w).sum(1)
    pos_diff_1 = torch.pow(pos_diff_1, 2)
    neg_diff_1 = torch.pow(neg_diff_1, 2)

    l1 = torch.mean(F.relu(pos_diff_1-neg_diff_1+0.5))

    pos_diff_2 = (PosN*w).sum(1)
    neg_diff_2 = (NegNp1*w).sum(1)
    pos_diff_2 = torch.pow(pos_diff_2, 2)
    neg_diff_2 = torch.pow(neg_diff_2, 2)
    l2 = torch.mean(F.relu(pos_diff_2-neg_diff_2+0.5))


    return (l1+l2)/2

def Self_Sparsity_Loss(x, element_logits,seq_len, n_similar, labels, device,args):
    spa_err=0
    for i in range(x.shape[0]):
        feat=x[i][:seq_len[i]]
        norm_feat=feat/(torch.norm(feat,p=2,dim=1,keepdim=True)+1e-8)
        sim_mat=torch.mm(norm_feat,norm_feat.permute(1,0))
        sim_mat=sim_mat-torch.eye(norm_feat.shape[0]).to(x.device)
        spa_err+=torch.sum(sim_mat)/(seq_len[i]*(seq_len[i]-1))

    return spa_err/x.shape[0]

def Cycle_Classification_Loss(x, element_logits,seq_len, n_similar, labels, device,args):
    cycle_loss=0

    for i in range(0,n_similar*2,1):

        feat_0=x[i][:seq_len[i]]
        feat_1=x[i+1][:seq_len[i+1]]
        # [n,m,c]
        feat_dis=(feat_0.unsqueeze(1)-feat_1.unsqueeze(0))
        feat_dis=torch.norm(feat_dis,p=2,dim=-1)**2
        # [n,m]
        alpha=torch.softmax(-feat_dis,dim=-1)
        # [n,m]*[m,c] -> [n,c]
        weighted_v=torch.mm(alpha,feat_1)

        logits=torch.softmax(-torch.norm(weighted_v.unsqueeze(1)-feat_0.unsqueeze(0),p=2,dim=-1)**2/0.5,dim=-1)
        target=torch.eye(seq_len[i]).to(x.device)
        cycle_loss+=torch.mean(-target*torch.log(logits+1e-8))
    return cycle_loss/n_similar


def Cycle_Regression_Loss(x,element_logits,seq_len,n_similar,labels, device,args):
    pass

def Cycle_Score_Loss(x, element_logits,seq_len, n_similar, labels, device,args):
    cycle_loss=0

    for i in range(0,n_similar*2,1):
        if i%2==0:
            feat_0=x[i][:seq_len[i]]
            feat_1=x[i+1][:seq_len[i+1]]
        else:
            feat_0=x[i][:seq_len[i]]
            feat_1=x[i-1][:seq_len[i-1]]
        # feat_0=feat_0/torch.norm(feat_0,p=2,dim=-1,keepdim=True)
        # feat_1=feat_1/torch.norm(feat_1,p=2,dim=-1,keepdim=True)
        # [n,m,c]

        feat_dis=(feat_0.unsqueeze(1)-feat_1.unsqueeze(0))
        feat_dis=torch.norm(feat_dis,p=2,dim=-1)**2
        # [n,m]
        alpha=torch.softmax(-feat_dis,dim=-1)
        # [n,m]*[m,c] -> [n,c]
        weighted_v=torch.mm(alpha,feat_1)

        # [n,n]
        weights=torch.softmax(-torch.norm(weighted_v.unsqueeze(1)-feat_0.unsqueeze(0),p=2,dim=-1)**2,dim=-1)
        # [n,n]->[n,n]*[n,c]
        logits=element_logits[i][:seq_len[i]]
        weights_logits=torch.mm(weights,logits)
        # just select the labels
        cycle_loss+=torch.mean((weights_logits.detach()-logits)**2)
        # cycle_loss+=torch.sum(torch.mm((weights_logits.detach()-logits)**2,(labels[i]*labels[i+1]).unsqueeze(-1)))/\
        #                       torch.sum(labels[i]*labels[i+1])
    return cycle_loss/n_similar

def Sparsity_Background_Loss(x,element_logits,seq_len,batch_size,labels,device,args):
    logits=element_logits
    background_logits=element_logits[:,:,-1]
    back_loss=0
    for i in range(logits.shape[0]):
        back_loss+=torch.mean(background_logits[i][:seq_len[i]])
    return back_loss/background_logits.shape[0]

def Norm_Loss(element_logits,seq_len,batch_size,labels,device,args):
    norm_loss=0
    element_logits=element_logits.squeeze(-1)
    for i in range(element_logits.shape[0]):
        norm_loss+=torch.norm(element_logits[i][:seq_len[i]],p=1,dim=0)
    norm_loss/=element_logits.shape[0]
    return norm_loss

def Cycle_Rerank_Loss(x,element_logits,seq_len,n_similar,labels,device,args):
    cycle_loss=0

    for i in range(0,n_similar*2,2):
        feat_0=x[i][:seq_len[i]]
        feat_1=x[i+1][:seq_len[i+1]]

        # we assump that

def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        print('using GPU to compute original distance')
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distmat.addmm_(1,-2,feat,feat.t())
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def MILL_Att(element_logits,att_scores, seq_len, batch_size, labels, device, args):
    ''' element_logits should be torch tensor of dimension (B, n_element, n_class),
         k should be numpy array of dimension (B,) indicating the top k locations to average over,
         labels should be a numpy array of dimension (B, n_class) of 1 or 0
         return is a torch tensor of dimension (B, n_class) '''

    k = np.ceil(seq_len/args.k).astype('int32')
    b, n_class = labels.shape
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    att_scores=att_scores.unsqueeze(-1)
    instance_logits = torch.zeros(0).to(device)
    for i in range(element_logits.shape[0]):
        logits=torch.relu(element_logits[i][:seq_len[i]])
        att=att_scores[i][:seq_len[i]]
        att_logits=att*logits
        tmp,_=torch.topk(att_logits,k=int(k[i]), dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

        # instance_logits.append(torch.mean(tmp, 0, keepdim=True)], dim=0))
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

# not work
def Vid_Feat_Guide_Loss(x,element_logits,att_feats,seq_len,labels,device,args):
    vg_sim_loss=0
    for i in range(element_logits.shape[0]):
        # [n,C]
        feat=x[i][:seq_len[i]]
        # element_logits [b,n,c] // labels [b,c]
        logits=torch.sigmoid(element_logits[i][:seq_len[i]])
        # [n,C]
        cls_scores=logits/torch.sum(logits,dim=0,keepdim=True)*labels[i].unsqueeze(0).repeat([logits.shape[0],1])
        # [n,C]*[n,c]
        weighted_feat=torch.sum(cls_scores.unsqueeze(-1)*feat.unsqueeze(1),dim=(0,1))/torch.sum(labels[i])
        vg_sim_loss+=torch.mean((weighted_feat.detach()-att_feats[i])**2)

    return vg_sim_loss/element_logits.shape[0]

def Mutual_Attn_Cls_Loss(element_logits,attns,seq_len,labels,device,args):
    def kl_div(input,target,eps=1e-8):
        return -torch.mean(target*torch.log(input+eps))
    mut_loss=0
    for i in range(element_logits.shape[0]):
        logits=torch.sigmoid(element_logits[i][:seq_len[i]])
        # [n,C]
        logits=torch.max(logits*labels[i].unsqueeze(0).repeat([logits.shape[0],1]),dim=1)[0]
        # logits and atten js div loss
        mut_loss+=0.5*kl_div(logits,0.5*(attns[i][:seq_len[i]]+logits))\
                  +0.5*kl_div(attns[i][:seq_len[i]],0.5*(logits+attns[i][:seq_len[i]]))
    return mut_loss/element_logits.shape[0]

def Attention_Guide_Loss(element_logits,att_scores,seq_len,batch_size,labels,device,args):
    # guide attention score via the variance of element logits
    AG_loss=0
    num_class=element_logits.shape[-1]
    pseudo_labels=torch.zeros(num_class,dtype=torch.float32)
    pseudo_labels[0]=1
    # max 0.05
    max_variance=torch.var(pseudo_labels,dim=0).to(element_logits.device)
    # min 0
    min_variance=0
    for i in range(element_logits.shape[0]):
        scores=torch.softmax(element_logits[i][:seq_len[i]],dim=-1)
        att=att_scores[i][:seq_len[i]]

        scores_var=torch.var(scores,dim=-1,keepdim=True)/max_variance
        AG_loss+=torch.mean((scores_var.detach()-att)**2)
    AG_loss/=element_logits.shape[0]
    return AG_loss

def Distance_Attention_Guide_Loss(x,element_logits,att_scores,weights,seq_len,device,args):
    # the similarity is define by cosine similarity
    DAG_loss=0
    for i in range(element_logits.shape[0]):
        # [n,c]
        feat=x[i][:seq_len[i]]
        att=att_scores[i][:seq_len[i]]
        # [n,c]*[c,C]->[n,C]
        cos_dis=torch.mm(feat,weights.permute(1,0))/torch.mm(torch.norm(feat,p=2,dim=1,keepdim=True),torch.norm(weights.permute(1,0),p=2,dim=0,keepdim=True))

def Attention_Cls_Loss(element_logits,att_scores,seq_len,labels,device,args):
    AC_loss=0
    for i in range(element_logits.shape[0]):
        scores=torch.sigmoid(element_logits[i][:seq_len[i]])
        sum_scores=torch.sum(scores*labels[i].unsqueeze(0),dim=-1)
        AC_loss+=torch.max(torch.abs(sum_scores-att_scores[i][:seq_len[i]].squeeze()),dim=0)[0]
    return AC_loss/element_logits.shape[0]

