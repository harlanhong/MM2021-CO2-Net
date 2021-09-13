import pandas as pd
import pdb
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import savgol_filter

def smooth(v):
   # return v
    l = min(200, len(v))
    l = l - (1-l%2)
    if len(v) <= 3:
        return v
    return savgol_filter(v, l, 2) #savgol_filter(v, l, 1) #0.5*(np.concatenate([v[1:],v[-1:]],axis=0) + v)

def task1():
    prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('temp/videl_list.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        pred = prediction[i]
        vn = videoname[i]
        gt = groundTruthGroupByVideoId.get_group(vn)
        if pred.shape[0]<10:
            count+=1
        continue
        for c in range(pred.shape[1]):
            cpred = pred[:,c]
            scpred = smooth(cpred)
            x = list(range(len(cpred)))
            cgt = gt[gt.label==c]
            gtline = np.array([np.min(cpred) for i in range(len(cpred))])
            for idx, this_pred in cgt.iterrows():
                gtline[this_pred['t-start']:this_pred['t-end']] = np.max(cpred)
            if len(cgt) == 0:
                continue
            plt.plot(x, cpred, linewidth=1)
            plt.plot(x, scpred, linewidth=1)
            threshold = np.max(scpred) - (np.max(scpred) - np.min(scpred)) * 0.05

            threshold = np.mean(cpred)
            plt.plot(x, gtline, linewidth=1)
            plt.plot(x, [threshold for i in range(len(cpred))], linewidth=1)
            
            plt.xlabel("clips", fontsize=12)
            plt.ylabel("scores", fontsize=12)
            plt.tick_params(axis='both',labelsize=10)
            plt.title("{} prediction".format(vn),fontsize=20)
            plt.savefig('temp/score_visual/{}_{}.pdf'.format(vn,c))
            print('save temp/score_visual/{}_{}.pdf'.format(vn,c))
            plt.clf()

    print(len(prediction),count)
if __name__ == '__main__':
    task1()
