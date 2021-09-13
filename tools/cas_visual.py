import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter
import torch
import cv2
def smooth(v):
   # return v
    l = min(10, len(v))
    l = l - (1-l%2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 4) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def cas_visulaize():
    # plt.axis('off')
    prediction = np.load('temp/model5.npy',allow_pickle=True).tolist()
    # videoname = np.load('temp/videl_list.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
        vn = videoname[i].decode()
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        pred = prediction[videoname[i]]['cas'][0]
        pred = torch.softmax(pred,-1).cpu().numpy()
        # pdb.set_trace()
        for c in range(pred.shape[1]):
            
            cpred = pred[:,c]
            # scpred = smooth(cpred)
            x = list(range(len(cpred)))
            # plt.plot(x, cpred, linewidth=1)
            plt.plot(x, cpred,label=str(c))
            # plt.xlabel("clips", fontsize=12)
            # plt.ylabel("scores", fontsize=12)
            # plt.tick_params(axis='both',labelsize=10)
            cgt = gt[gt.label==c]
            if len(cgt) == 0:
                continue
            gtline = np.array([0 for i in range(len(cpred))])
            for idx, this_pred in gt.iterrows():
                gtline[int(this_pred['t-start']*30/16):int(this_pred['t-end']*30/16)] = 1
            plt.plot(x, gtline,label='gt_'+str(c))
        
        plt.title("{} {}".format(vn,','.join([str(i) for i in np.unique(gt['label'].tolist())])),fontsize=20)
        plt.legend()
        plt.savefig('temp/score_visual/{}.pdf'.format(vn))
        print('save temp/score_visual/{}_{}.pdf'.format(vn,c))
        plt.clf()

    print(len(prediction),count)
if __name__ == '__main__':
    cas_visulaize()