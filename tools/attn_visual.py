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


#ours方法和ours_wo_attn方法的attn可视化
def attn_compared():
    plt.axis('off')
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    attns_wo_neg = np.load('temp/attn_wo_atnn.npy',allow_pickle=True).tolist()
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        # if not '0541' in vn:
        #     continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        attn_sigmoid,attn_logit = attns[videoname[i]]
        attn_sigmoid_wo_neg,attn_logit_wo_neg = attns_wo_neg[videoname[i]]

        attn = attn_sigmoid.view(-1,).cpu().numpy()
        attn_wo_neg = attn_sigmoid_wo_neg.view(-1,).cpu().numpy()

        # attn_softmax = torch.softmax(attn_logit,1)
        # attn_softmax_wo_neg = torch.softmax(attn_logit_wo_neg,1)
        # attn = attn_softmax.view(-1,).cpu().numpy()
        # attn_wo_neg = attn_softmax_wo_neg.view(-1,).cpu().numpy()


        # pdb.set_trace()
        x = list(range(len(attn)))
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*fps/16):int(this_pred['t-end']*fps/16)] = np.max(attn)
        
        # plt.clf()
        attn = smooth(attn)
        attn_wo_neg = smooth(attn_wo_neg)

        plt.plot(x, attn, linewidth=1,color='green')
        plt.plot(x, attn_wo_neg, linewidth=1,color='blue')

        # plt.fill_between(x, attn_sigmoid, interpolate=True, linewidth=1, color='green', alpha=0.3)
        plt.plot(x, gtline, linewidth=1,color='red')
        plt.savefig('temp/attn/{}.pdf'.format(vn))

        print('save temp/attn/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)

#framework中attn的直方图
def attn_visualize():
    plt.axis('off')
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    # attns_wo_neg = np.load('temp/attn_wo_neg.npy',allow_pickle=True).tolist()
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()

        if 'validation' in vn:
            continue
        attn_sigmoid,attn_logit = attns[videoname[i]]
        attn = attn_sigmoid.view(-1,).cpu().numpy()
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        gt = groundTruthGroupByVideoId.get_group(vn)
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*30/16):int(this_pred['t-end']*30/16)] = np.max(attn)
        
        # pdb.set_trace()
        x = list(range(len(attn)))
        
        # plt.clf()
        # attn = smooth(attn)
        # pdb.set_trace()
        
        attn = smooth(attn)

        # plt.plot(x, attn, linewidth=1,color='green')
        plt.bar(x, attn,0.7,color=['r','g','b'],alpha=0.5)
        plt.plot(x, gtline, linewidth=1,color='red')
        # plt.plot(x, gtline, linewidth=1,color='red')
        # plt.title("{} prediction".format(vn),fontsize=20)
        plt.savefig('temp/attn/{}.pdf'.format(vn))
        print('save temp/attn/{}.pdf'.format(vn))
        plt.clf()
        plt.axis('off')

    # print(len(prediction),count)

def logit_visulaize():
    plt.axis('off')
    prediction = np.load('temp/logits.npy',allow_pickle=True).tolist()
    # videoname = np.load('temp/videl_list.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
        vn = videoname[i].decode()
        if 'validation' in vn:
            continue
        pred = prediction[videoname[i]][0]
        pred = torch.softmax(pred,-2)
        # pdb.set_trace()
        for c in range(pred.shape[1]):
            cpred = pred[:,c]
            scpred = smooth(cpred)
            x = list(range(len(cpred)))
            # plt.plot(x, cpred, linewidth=1)
            plt.bar(x, cpred)
            # plt.xlabel("clips", fontsize=12)
            # plt.ylabel("scores", fontsize=12)
            # plt.tick_params(axis='both',labelsize=10)
            plt.title("{} prediction".format(vn),fontsize=20)
            plt.savefig('temp/score_visual/{}_{}.pdf'.format(vn,c))
            print('save temp/score_visual/{}_{}.pdf'.format(vn,c))
            plt.clf()

    print(len(prediction),count)

#放到framework中的attn和T-CAMs
def attn_TCAMs_visualize():
    plt.axis('off')
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    # attns_wo_neg = np.load('temp/attn_wo_neg.npy',allow_pickle=True).tolist()
    prediction = np.load('temp/logits.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    colors = ['dodgerblue','darkviolet','goldenrod','firebrick','yellow']
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        if not '0000846' in vn:
            continue
        if 'validation' in vn:
            continue
        attn_sigmoid,attn_logit = attns[videoname[i]]
        pred = prediction[videoname[i]][0]
        pred = torch.softmax(pred,-1)
        attn = attn_sigmoid.view(-1,).cpu().numpy()
        x = list(range(len(attn)))
        filtered_pred = attn.reshape(-1,1)*pred.numpy()
        print(len(x))
        begin = 0
        end=len(x)
        # pdb.set_trace()

        # plt.plot(x, attn, linewidth=1,color='green')
        plt.bar(x[begin:end], attn[begin:end],0.7,color=['g'],alpha=0.5)
        # colors = ['']
        plt.axis('off')
        plt.savefig('temp/framework/attn.pdf')
        plt.clf()
        for c in range(pred.shape[1]):
            cpred = pred[:,c]
            fcpred = filtered_pred[:,c]

            # scpred = smooth(cpred)
            x = list(range(len(cpred)))
            
            plt.bar(x[begin:end], cpred[begin:end],0.7,color=colors[c%4], linewidth=1,alpha=0.5)
            
            plt.axis('off')
            plt.savefig('temp/framework/{}.pdf'.format(c))
            plt.clf()

            plt.bar(x[begin:end], fcpred[begin:end],0.7,color=colors[c%4], linewidth=1,alpha=0.5)
            plt.axis('off')
            plt.savefig('temp/framework/filtered_{}.pdf'.format(c))
            plt.clf()

            print('save temp/framework/{}.pdf'.format(c))
        print('save temp/attn/{}.pdf'.format(vn))
        

    # print(len(prediction),count)

def task2():
    plt.axis('off')
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    attns_wo_neg = np.load('temp/attn_wo_neg.npy',allow_pickle=True).tolist()
    logits = np.load('temp/logits.npy',allow_pickle=True).tolist()
    logits_wo_neg = np.load('temp/logits_wo_neg.npy',allow_pickle=True).tolist()

    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0

    for i in range(len(videoname)):
        vn = videoname[i].decode()
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        gt_label = np.unique(gt['label'].tolist())
        if len(gt_label)>1: #暂时先只探究一个
            continue
        attn_sigmoid, attn_logit = attns[videoname[i]]
        attn_sigmoid_wo_neg, attn_logit_wo_neg = attns_wo_neg[videoname[i]]
        pred = attn_sigmoid_wo_neg*logits[videoname[i]]
        pred_wo_neg = attn_sigmoid_wo_neg*logits_wo_neg[videoname[i]]

        pred = pred.cpu().numpy()[0][:,gt_label[0]]
        pred_wo_neg = pred_wo_neg.cpu().numpy()[0][:,gt_label[0]]

        # pdb.set_trace()
        x = list(range(len(pred)))
        
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(pred) for i in range(len(pred))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*25/16):int(this_pred['t-end']*25/16)] = np.max(pred)
        
        # plt.clf()

        plt.plot(x, pred, linewidth=1,color='green')
        plt.plot(x, pred_wo_neg, linewidth=1,color='blue')

        # plt.fill_between(x, attn_sigmoid, interpolate=True, linewidth=1, color='green', alpha=0.3)

        # plt.fill_between(x, attn_sigmoid_wo_neg, interpolate=True, linewidth=1, color='blue', alpha=0.3)
        
        plt.plot(x, gtline, linewidth=1,color='red')
        
        # plt.xlabel("clips", fontsize=12)
        # plt.ylabel("scores", fontsize=12)
        # plt.tick_params(axis='both',labelsize=10)
        # plt.title("{} prediction".format(vn),fontsize=20)
        plt.savefig('temp/score_visual/{}.pdf'.format(vn))

        print('save temp/score_visual/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


#画wo_neg对比图
def task3():
    plt.axis('off')
    attns = np.load('temp/attn.npy',allow_pickle=True).tolist()
    attns_wo_neg = np.load('temp/attn_wo_neg.npy',allow_pickle=True).tolist()
    
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        if not '541' in vn:
            continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        attn_sigmoid,attn_logit = attns[videoname[i]]
        attn_sigmoid_wo_neg,attn_logit_wo_neg = attns_wo_neg[videoname[i]]

        attn = attn_sigmoid.view(-1,).cpu().numpy()
        attn_wo_neg = attn_sigmoid_wo_neg.view(-1,).cpu().numpy()

        # pdb.set_trace()
        x = list(range(len(attn)))
        
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*30/16):int(this_pred['t-end']*30/16)] = np.max(attn)
        
        # plt.clf()
        attn = smooth(attn)
        attn_wo_neg = smooth(attn_wo_neg)

        # plt.plot(x, attn, linewidth=1,color='green')
        plt.fill_between(x, attn, interpolate=True, linewidth=1, color='green', alpha=0.3)

        plt.savefig('temp/score_visual/{}.pdf'.format(vn))
        plt.clf()
        plt.axis('off')
        # plt.plot(x, attn_wo_neg, linewidth=1,color='blue')
        plt.fill_between(x, attn_wo_neg, interpolate=True, linewidth=1, color='blue', alpha=0.3)
        plt.savefig('temp/score_visual/{}_wo_neg.pdf'.format(vn))


        
        # plt.plot(x, gtline, linewidth=1,color='red')

        print('save temp/score_visual/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


def task5():
    plt.axis('off')
    attns=np.load('../temp/thumos_prediction_ours.npy',allow_pickle=True).tolist()
    # attns=results['attn']
    # prediction=results['cas']
    # attns = np.load('temp/attn.npy', allow_pickle=True).tolist()
    # attns_wo_neg = np.load('temp/attn_wo_neg.npy', allow_pickle=True).tolist()

    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',
                        allow_pickle=True).tolist()
    groundTruth = pd.read_csv('../temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')

    count = 0
    for i in range(len(videoname)):
        # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        # if not '541' in vn:
        #     continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        # attn_sigmoid = attns[videoname[i]]
        res=attns[videoname[i]]
        vatt=res['v_attn'].view(-1,).cpu().numpy()
        fatt=res['f_attn'].view(-1,).cpu().numpy()
        attn=res['attn'].view(-1,).cpu().numpy()
        # attn_sigmoid_wo_neg, attn_logit_wo_neg = attns_wo_neg[videoname[i]]

        # attn = attn_sigmoid.view(-1, ).cpu().numpy()
        # attn_wo_neg = attn_sigmoid_wo_neg.view(-1, ).cpu().numpy()

        # pdb.set_trace()
        x = list(range(len(attn)))

        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start'] * 30 / 16):int(this_pred['t-end'] * 30 / 16)] = np.max(attn)

        # plt.clf()
        attn = smooth(attn)
        # attn_wo_neg = smooth(attn_wo_neg)

        # plt.plot(x, attn, linewidth=1,color='green')
        plt.fill_between(x, attn, interpolate=True, linewidth=1, color='green', alpha=0.3)

        plt.savefig('../temp/attn_vis/{}.pdf'.format(vn))
        plt.clf()
        plt.axis('off')

        # plt.plot(x, attn_wo_neg, linewidth=1,color='blue')

        plt.fill_between(x, vatt, interpolate=True, linewidth=1, color='blue', alpha=0.3)
        plt.savefig('../temp/attn_vis/{}_vatt.pdf'.format(vn))
        plt.clf()
        plt.axis('off')

        plt.fill_between(x, fatt, interpolate=True, linewidth=1, color='purple', alpha=0.3)
        plt.savefig('../temp/attn_vis/{}_fatt.pdf'.format(vn))
        # plt.plot(x, gtline, linewidth=1,color='red')

        print('save temp/attn_vis/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


#draw_video
def task4():
   

    cap = cv2.VideoCapture()  #读取摄像头
    #cap = cv2.VideoCapture("video.mp4")  #读取视频文件
    fps = 15
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc('M','P','E','G'), fps, size)

    while(True):
        ret, frame = cap.read()
        if ret:
            videoWriter.write(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    videoWriter.release()


#attn可视化 mm2021
def attn_mm():
    plt.axis('off')
    results = np.load('temp/thumos_prediction.npy',allow_pickle=True).tolist()
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        # if not '0541' in vn:
        #     continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        attn = results[videoname[i]]['attn']
        f_attn = results[videoname[i]]['f_attn']
        v_attn = results[videoname[i]]['v_attn']
        attn = attn.view(-1,).cpu().numpy()
        f_attn = f_attn.view(-1,).cpu().numpy()
        v_attn = v_attn.view(-1,).cpu().numpy()

        # pdb.set_trace()
        x = list(range(len(attn)))
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*25/16):int(this_pred['t-end']*25/16)] = np.max(attn)
        
        # plt.clf()
        # attn = smooth(attn)
        # attn_wo_neg = smooth(attn_wo_neg)

        plt.plot(x, attn, linewidth=1,color='green')
        plt.plot(x, f_attn, linewidth=1,color='blue')
        plt.plot(x, v_attn, linewidth=1,color='black')

        # plt.fill_between(x, attn_sigmoid, interpolate=True, linewidth=1, color='green', alpha=0.3)
        plt.plot(x, gtline, linewidth=1,color='red')
        plt.savefig('temp/attn/{}.pdf'.format(vn))
        print('save temp/attn/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


def attn_mm_select():
    plt.axis('off')
    results = np.load('temp/thumos_prediction.npy',allow_pickle=True).tolist()
    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        if not '0450' in vn:
            continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        attn = results[videoname[i]]['attn']
        f_attn = results[videoname[i]]['f_attn']
        v_attn = results[videoname[i]]['v_attn']
        attn = attn.view(-1,).cpu().numpy()
        f_attn = f_attn.view(-1,).cpu().numpy()
        v_attn = v_attn.view(-1,).cpu().numpy()

        # pdb.set_trace()
        x = list(range(len(attn)))
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(attn) for i in range(len(attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*25/16):int(this_pred['t-end']*25/16)] = np.max(attn)
        
        # plt.clf()
        # attn = smooth(attn)
        # attn_wo_neg = smooth(attn_wo_neg)
        plt.clf()
        plt.axis('off')
        plt.bar(x, attn, linewidth=1,color='forestgreen')
        plt.savefig('temp/framework/{}_attn.pdf'.format(vn))
        plt.clf()
        plt.axis('off')
        plt.bar(x, f_attn, linewidth=1,color='lightsalmon')
        plt.savefig('temp/framework/{}_f_attn.pdf'.format(vn))
        plt.clf()
        plt.axis('off')
        plt.bar(x, v_attn, linewidth=1,color='firebrick')
        plt.savefig('temp/framework/{}_v_attn.pdf'.format(vn))

        # plt.fill_between(x, attn_sigmoid, interpolate=True, linewidth=1, color='green', alpha=0.3)
        # plt.plot(x, gtline, linewidth=1,color='red')
        # plt.savefig('temp/attn/{}.pdf'.format(vn))
        print('save temp/attn/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)

def attn_compare():
    plt.axis('off')
    cem = np.load('temp/Model45_3552_full_seq_500.npy',allow_pickle=True).tolist()
    add = np.load('temp/Model45_Add_3552_full.npy',allow_pickle=True).tolist()
    concate = np.load('temp/Model45_3552_no_ch_bit_conv.npy',allow_pickle=True).tolist()
    ssma = np.load('temp/Model45_SSMA_3552_full.npy',allow_pickle=True).tolist()

    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        # if not '0450' in vn:
        #     continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        pdb.set_trace()
        cem_attn = cem[videoname[i]]['attn'][0]
        add_attn = add[videoname[i]]['attn'][0]
        ssma_attn = ssma[videoname[i]]['attn'][0]
        concate_attn = concate[videoname[i]]['attn'][0]

        cem_attn = cem_attn.view(-1,).cpu().numpy()
        add_attn = add_attn.view(-1,).cpu().numpy()
        ssma_attn = ssma_attn.view(-1,).cpu().numpy()
        concate_attn = concate_attn.view(-1,).cpu().numpy()


        # pdb.set_trace()
        x = list(range(len(cem_attn)))
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(cem_attn) for i in range(len(cem_attn))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*25/16):int(this_pred['t-end']*25/16)] = np.max(cem_attn)
        
        # plt.clf()
        cem_attn = smooth(cem_attn)
        concate_attn = smooth(concate_attn)
        plt.clf()
        # plt.axis('off')
        plt.plot(list(range(len(cem_attn))), cem_attn, linewidth=1,color='forestgreen')
        # plt.plot(list(range(len(add_attn))), add_attn, linewidth=1,color='blue')
        # plt.plot(list(range(len(ssma_attn))), ssma_attn, linewidth=1,color='black')
        plt.plot(list(range(len(concate_attn))), concate_attn, linewidth=1,color='pink')

        plt.plot(x, gtline, linewidth=1,color='red')

        plt.savefig('temp/attn/{}_attn.pdf'.format(vn))
        
        # plt.savefig('temp/attn/{}.pdf'.format(vn))
        print('save temp/attn/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


def cas_compare():
    plt.axis('off')
    cem = np.load('temp/Model45_3552_full_seq_500.npy',allow_pickle=True).tolist()
    add = np.load('temp/Model45_Add_3552_full.npy',allow_pickle=True).tolist()
    concate = np.load('temp/Model45_concate_3553.npy',allow_pickle=True).tolist()
    ssma = np.load('temp/Model45_SSMA_3552_full.npy',allow_pickle=True).tolist()
    se = np.load('temp/Model45_se_3552.npy',allow_pickle=True).tolist()

    # prediction = np.load('temp/prediction.npy',allow_pickle=True).tolist()
    videoname = np.load('./Thumos14reduced-Annotations/videoname.npy',allow_pickle=True).tolist()
    groundTruth = pd.read_csv('temp/groundtruth.csv')
    groundTruthGroupByVideoId = groundTruth.groupby('video-id')
    count = 0
    for i in range(len(videoname)):
    # for i in range(2):
        # pred = prediction[i]
        vn = videoname[i].decode()
        if not vn in ['video_test_0000062','video_test_0000450','video_test_0000519','video_test_0000541','video_test_0001276']:
            continue
        if 'validation' in vn:
            continue
        gt = groundTruthGroupByVideoId.get_group(vn)
        labels = np.unique(gt['label'].tolist())
        labindx = 0
        if len(labels)>1:
            continue
        else:
            labindx = labels[0]

        cem_cas = torch.softmax(cem[videoname[i]]['cas'][0],-1)[:,labindx]
        add_cas = torch.softmax(add[videoname[i]]['cas'][0],-1)[:,labindx]
        ssma_cas = torch.softmax(ssma[videoname[i]]['cas'][0],-1)[:,labindx]
        concate_cas = torch.softmax(concate[videoname[i]]['cas'][0],-1)[:,labindx]
        se_cas = torch.softmax(se[videoname[i]]['cas'][0],-1)[:,labindx]
        
        cem_cas = cem_cas.view(-1,).cpu().numpy()
        add_cas = add_cas.view(-1,).cpu().numpy()
        ssma_cas = ssma_cas.view(-1,).cpu().numpy()
        concate_cas = concate_cas.view(-1,).cpu().numpy()
        se_cas = se_cas.view(-1,).cpu().numpy()


        # pdb.set_trace()
        x = list(range(len(cem_cas)))
        video = cv2.VideoCapture('/home/share2/jiachang/TH14_test_set_mp4/'+vn+'.mp4')
        fps = video.get(cv2.CAP_PROP_FPS)
        # cgt = gt[gt.label==c]
        gtline = np.array([np.min(cem_cas) for i in range(len(cem_cas))])
        for idx, this_pred in gt.iterrows():
            gtline[int(this_pred['t-start']*25/16):int(this_pred['t-end']*25/16)] = np.max(cem_cas)
        
        # plt.clf()
        # cem_cas = smooth(cem_cas)
        # concate_cas = smooth(concate_cas)
        plt.clf()
        plt.axis('off')
        # plt.plot(list(range(len(cem_cas))), cem_cas, linewidth=3, color='dodgerblue')
        plt.fill_between(list(range(len(cem_cas))), cem_cas, interpolate=True, linewidth=1, alpha=0.5)
        plt.savefig('temp/score_visual/{}_cem.pdf'.format(vn))
        plt.clf()
        plt.axis('off')
        # plt.plot(list(range(len(add_cas))), add_cas, linewidth=3, color='dodgerblue')
        plt.fill_between(list(range(len(add_cas))), add_cas, interpolate=True, linewidth=1, alpha=0.5)
        plt.savefig('temp/score_visual/{}_add.pdf'.format(vn))
        plt.clf()
        plt.axis('off')
        # plt.plot(list(range(len(ssma_cas))), ssma_cas, linewidth=3, color='dodgerblue')
        plt.fill_between(list(range(len(ssma_cas))), ssma_cas, interpolate=True, linewidth=1, alpha=0.5)
        plt.savefig('temp/score_visual/{}_ssma.pdf'.format(vn))

        plt.clf()
        plt.axis('off')
        # plt.plot(list(range(len(concate_cas))), concate_cas, linewidth=3, color='dodgerblue')
        plt.fill_between(list(range(len(concate_cas))), concate_cas, interpolate=True, linewidth=1, alpha=0.5)
        plt.savefig('temp/score_visual/{}_concate.pdf'.format(vn))

        plt.clf()
        plt.axis('off')
        # plt.plot(list(range(len(concate_cas))), concate_cas, linewidth=3, color='dodgerblue')
        plt.fill_between(list(range(len(se_cas))), se_cas, interpolate=True, linewidth=1, alpha=0.5)
        plt.savefig('temp/score_visual/{}_se.pdf'.format(vn))

        # plt.plot(x, gtline, linewidth=1,color='red')

        # plt.savefig('temp/score_visual/{}_attn.pdf'.format(vn))
        
        # plt.savefig('temp/attn/{}.pdf'.format(vn))
        print('save temp/score_visual/{}.pdf'.format(vn))
        plt.clf()

    # print(len(prediction),count)


if __name__ == '__main__':
    # task1()a
    # attn_mm_select()
    cas_compare()