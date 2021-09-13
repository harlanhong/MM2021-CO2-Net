import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
import torch
import cv2
import matplotlib as mlp
from matplotlib.pyplot import MultipleLocator
mlp.rcParams['axes.spines.right'] = False
mlp.rcParams['axes.spines.top'] = False
def smooth(v):
   # return v
    l = min(10, len(v))
    l = l - (1-l%2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 4) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

#分值压制可视化，full method
def task5():
    plt.axis('off')

    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    logtis = np.load('temp/logits.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
        vn = videoname[i].decode()
        if not '0000026' in vn:
            continue
        if 'validation' in vn:
            continue
        pred = logtis[videoname[i]][0]
        atn,attn_logit = attns[videoname[i]]
        atn = atn.view(-1,1)
        _min = pred.min(dim=-1, keepdim=True)[0]
        pred = atn * (pred - _min) + _min

        pred_pro = torch.softmax(pred,-1).numpy()
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        gt = groundTruthGroupByVideoId.get_group(vn)
        gt_pred = []
        actions =[]
        for idx, this_pred in gt.iterrows():
            gt_pred.append(pred_pro[int(this_pred['t-start']*fps/16):int(this_pred['t-end']*fps/16)])
            actions.append(this_pred['label'])
        actions = np.unique(actions)
        gt_pred = np.concatenate(gt_pred,0)
        pred_pro_rows = pred_pro.view([('', pred_pro.dtype)] * pred_pro.shape[1])
        gt_pred_rows = gt_pred.view([('', gt_pred.dtype)] * gt_pred.shape[1])

        bg_pro=np.setdiff1d(pred_pro_rows, gt_pred_rows).view(pred_pro.dtype).reshape(-1, pred_pro.shape[1])
        # pdb.set_trace()
        x = list(range(pred.shape[-1]))

        gt_pred = gt_pred.mean(0).reshape(-1)
        bg_pro = bg_pro.mean(0).reshape(-1)
        pred_pro = pred_pro.mean(0).reshape(-1)

        gt_ctg = np.zeros(len(x))
        for i in actions:
            gt_ctg[i] = 1

        red_gt = [0]*len(gt_pred)
        for i in range(len(gt_pred)):
            if i in actions:
                red_gt[i] = gt_pred[i]
                gt_pred[i] = 0
            else:
                red_gt[i] = 0
        plt.clf()
        #fg图
        plt.grid(linestyle='-.')
        plt.bar(x, gt_pred.tolist(),0.7,color=['r'],alpha=0.5)
        plt.bar(x, red_gt,0.7,color=['g'],alpha=0.5)

        # plt.bar(x, gt_ctg.tolist(),0.5,color=['b'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/suppression/{}_fg.pdf'.format(vn))
        '''
        #bg图
        plt.clf()
        # plt.axis('off')
        plt.grid(linestyle='-.')
        plt.bar(x, bg_pro.tolist(),0.7,color=['r'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/score_visual/{}_bg.pdf'.format(vn))
        #all图
        plt.clf()
        plt.grid(linestyle='-.')
        plt.bar(x, pred_pro.tolist(),0.7,color=['r'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/score_visual/{}_all.pdf'.format(vn))
        '''
        print('save temp/suppression/{}.pdf'.format(vn))


    # print(len(prediction),count)

#分值压制可视化，wo comlementary method
def task6():
    plt.axis('off')

    attns = np.load('temp/attn_wo_neg.npy',allow_pickle=True).tolist()
    logtis = np.load('temp/logits_wo_neg.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
        vn = videoname[i].decode()
        if not '0000026' in vn:
            continue
        if 'validation' in vn:
            continue
        # pred = logtis[videoname[i]][0]
        pred = logtis[videoname[i]][0]
        #========
        atn,attn_logit = attns[videoname[i]]
        atn = atn.view(-1,1)
        _min = pred.min(dim=-1, keepdim=True)[0]
        pred = atn * (pred - _min) + _min

        pred_pro = torch.softmax(pred,-1).numpy()
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        gt = groundTruthGroupByVideoId.get_group(vn)
        gt_pred = []
        actions =[]
        for idx, this_pred in gt.iterrows():
            gt_pred.append(pred_pro[int(this_pred['t-start']*fps/16):int(this_pred['t-end']*fps/16)])
            actions.append(this_pred['label'])
        actions = np.unique(actions)
        gt_pred = np.concatenate(gt_pred,0)
        pred_pro_rows = pred_pro.view([('', pred_pro.dtype)] * pred_pro.shape[1])
        gt_pred_rows = gt_pred.view([('', gt_pred.dtype)] * gt_pred.shape[1])

        bg_pro=np.setdiff1d(pred_pro_rows, gt_pred_rows).view(pred_pro.dtype).reshape(-1, pred_pro.shape[1])
        # x = list(range(pred.size[1]))
        x = list(range(pred.shape[-1]))

        gt_pred = gt_pred.mean(0).reshape(-1)
        bg_pro = bg_pro.mean(0).reshape(-1)
        pred_pro = pred_pro.mean(0).reshape(-1)

        gt_ctg = np.zeros(len(x))
        for i in actions:
            gt_ctg[i] = 1

        red_gt = [0]*len(gt_pred)
        for i in range(len(gt_pred)):
            if i in actions:
                red_gt[i] = gt_pred[i]
                gt_pred[i] = 0
            else:
                red_gt[i] = 0
        
        #fg图
        plt.clf()
        plt.grid(linestyle='-.')
        plt.bar(x, gt_pred.tolist(),0.7,color=['r'],alpha=0.5)
        plt.bar(x, red_gt,0.7,color=['g'],alpha=0.5)

        # plt.bar(x, gt_ctg.tolist(),0.5,color=['b'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/suppression/{}_fg_wo_neg.pdf'.format(vn))
        '''
        #bg图
        plt.clf()
        # plt.axis('off')
        plt.grid(linestyle='-.')
        plt.bar(x, bg_pro.tolist(),0.7,color=['r'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/score_visual/{}_bg_wo_neg.pdf'.format(vn))
        #all图
        plt.clf()
        plt.grid(linestyle='-.')
        plt.bar(x, pred_pro.tolist(),0.7,color=['r'],alpha=0.5)
        plt.xticks(x)
        plt.savefig('temp/score_visual/{}_all_wo_neg.pdf'.format(vn))
        '''
        print('save temp/suppression/{}_wo_neg.pdf'.format(vn))


    # print(len(prediction),count)

if __name__ == '__main__':
    task5()
    task6()