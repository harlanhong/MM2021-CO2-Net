import numpy as np
import pdb
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.metrics.pairwise import cosine_similarity
def vis_feat_dist_map(dist_map,path):
    plt.clf()
    # fig=plt.figure()
    # ax=fig.add_subplot(111)
    res=sn.heatmap(dist_map,vmin=0,vmax=1)
    figure=res.get_figure()
    figure.savefig(path,dpi=400)

if __name__ == '__main__':
    path_to_features = 'path/to/Thumos14reduced-I3D-JOINTFeatures.npy'
    path_to_labels = './Thumos14reduced-Annotations/labels_all.npy'
    path_to_segments = './Thumos14reduced-Annotations/segments.npy'

    # labels = np.load( + 'labels_all.npy',allow_pickle=True)     # Specific to Thumos14
    labels = np.load(path_to_labels, allow_pickle=True)     # Specific to Thumos14
    segments = np.load(path_to_segments,allow_pickle=True)
    features = np.load(path_to_features, encoding='bytes',allow_pickle=True)
    feats = features[0]
    rgb = feats[:,:1024]
    flow = feats[:,1024:]
    n,d = rgb.shape
    dist_map=np.empty([n,n])


    for i in range(n):
        for j in range(n):
            sim=cosine_similarity(rgb[i].reshape([1,-1]),rgb[j].reshape([1,-1]))
            dist_map[i][j]=sim.ravel()
    dist_map = dist_map - dist_map.min()
    dist_map /= dist_map.max()
    vis_feat_dist_map(dist_map,'2.jpg')
    pdb.set_trace()
    print('aa')
