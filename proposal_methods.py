
import numpy as np
import torch
import utils.wsad_utils as utils
from scipy.signal import savgol_filter
import pdb
import pandas as pd
import options
args = options.parser.parse_args()

def filter_segments(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        #s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

def filter_segments_2(segment_predict, vn):
    ambilist = './Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        #s[j], e[j], np.max(seg)+0.7*c_s[c],c]
        for a in ambilist:
            if a[0] == vn:
                if not (max(float(a[2]),segment_predict[i][0])>=min(float(a[3]),segment_predict[i][1])):
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)

def smooth(v, order=2,lens=200):
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)

def get_topk_mean(x, k, axis=0):
    return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)
def get_cls_score(element_cls, dim=-2, rat=20, ind=None):

    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] // rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score

def _get_vid_score(pred):
    # pred : (n, class)
    if args is None:
        k = 8
        topk_mean = self.get_topk_mean(pred, k)
        # ind = topk_mean > -50
        return pred, topk_mean

    win_size = int(args.topk)
    split_list = [i*win_size for i in range(1, int(pred.shape[0]//win_size))]
    splits = np.split(pred, split_list, axis=0)

    tops = []
    #select the avg over topk2 segments in each window
    for each_split in splits:
        top_mean = get_topk_mean(each_split, args.topk2)
        tops.append(top_mean)
    tops = np.array(tops)
    c_s = np.max(tops, axis=0)
    return pred, c_s
def __vector_minmax_norm(vector, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        max_val = np.max(vector)
        min_val = np.min(vector)

    delta = max_val - min_val
    # delta[delta <= 0] = 1
    ret = (vector - min_val) / delta

    return ret


def multiple_threshold(vid_name,instance_logit,bag_logit):
    # act_thresh_cas = np.arange(0.1, 0.25, 0.025)
    act_thresh_cas = np.linspace(0.1,0.9,10)

    num_segments = instance_logit.shape[1]

    score_np = torch.softmax(bag_logit,1).cpu().numpy()[0]
    pred = np.where(score_np >= 0.2)[0]
    if len(pred) == 0:
        pred = np.array([np.argmax(score_np)])
    cas_softmax = torch.softmax(instance_logit,-1)
    #归一化
    cas = utils.minmax_norm(cas_softmax)
    cas_pred = cas[0].cpu().numpy()[:, pred] #预测类的分值
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1)) #[num_segments,pred,1]
    # cas_pred = utils.upgrade_resolution(cas_pred, args.scale)
    
    proposal_dict = {}

    for i in range(len(act_thresh_cas)):
        cas_temp = cas_pred.copy()
        #查出低于阈值的
        zero_location = np.where(cas_temp[:, :, 0] < act_thresh_cas[i])
        cas_temp[zero_location] = 0
        #有效段
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp[:, c, 0] > 0)
            seg_list.append(pos)
        
        proposals = utils.get_proposal_oic(seg_list, cas_temp, score_np, pred)
        
        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[i]
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(utils.nms(proposal_dict[class_id], 0.6))
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    #[c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end,c_score,c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

def multiple_threshold_2(vid_name,instance_logit,bag_logit,cas_supp_atn=None):
    act_thresh_cas = np.arange(0.1, 0.9, 10)

    num_segments = instance_logit.shape[1]
    pred_vid_score = get_cls_score(instance_logit, rat=args.k)
    # prediction, c_s = _get_vid_score(instance_logit)

    score_np = pred_vid_score.copy()
    # score_np = torch.softmax(bag_logit,1).cpu().numpy()[0]
    # logit_atn = cas_supp_atn.expand_as(instance_logit).squeeze().data.cpu().numpy()

    pred = np.where(score_np >= 0.2)[0]
    if len(pred) == 0:
        pred = np.array([np.argmax(score_np)])
    cas_softmax = torch.softmax(instance_logit,-1)
    #归一化
    cas = utils.minmax_norm(cas_softmax)
    cas_pred = cas[0].cpu().numpy()[:, pred] #预测类的分值
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1)) #[num_segments,pred,1]
    # cas_pred = utils.upgrade_resolution(cas_pred, args.scale)
    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]
    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    for i in range(len(act_thresh_cas)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        #查出低于阈值的
        # zero_location = np.where(cas_temp[:, :, 0] < act_thresh_cas[i])
        # cas_temp[zero_location] = 0
        #有效段
        seg_list = []
        # for c in range(len(pred)):
        #     pos = np.where(cas_temp[:, c, 0] > 0)
        #     seg_list.append(pos)
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh_cas[i])
            seg_list.append(pos)
        
        proposals = utils.get_proposal_oic_2(seg_list, cas_temp, score_np, pred,args.scale,num_segments,args.feature_fps,num_segments,gamma=0.2)

        for i in range(len(proposals)):
            class_id = proposals[i][0][0]
            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []
            proposal_dict[class_id] += proposals[i]
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    #[c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end,c_score,c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments_2(segment_predict, vid_name.decode())
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

@torch.no_grad()
def multiple_threshold_hamnet(vid_name,data_dict):
    elem = data_dict['cas']
    element_atn=data_dict['attn']
    act_thresh_cas = np.arange(0.1, 0.9, 10)
    element_logits = elem * element_atn
    # element_logits = data_dict['cas_supp']
    pred_vid_score = get_cls_score(element_logits, rat=10)
    score_np = pred_vid_score.copy()
    # score_np[score_np < 0.2] = 0
    # score_np[score_np >= 0.2] = 1
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    pred = np.where(pred_vid_score >= 0.2)[0]

    # NOTE: threshold
    act_thresh = np.linspace(0.1,0.9,10)
    # act_thresh = np.linspace(0.2,0.4,10)
    
    prediction = None
    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    cas_pred = cas_supp[0].cpu().numpy()[:, pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]

    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))

    proposal_dict = {}

    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                            cas_temp,
                                            pred_vid_score,
                                            pred,
                                            args.scale,
                                            num_segments,
                                            args.feature_fps,
                                            num_segments,
                                            gamma=args.gamma_oic)

        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                logger.error(f"Index error")
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
    # self.final_res["results"][vid_name[0]] = utils.result2json(
        # final_proposals, class_dict)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    #[c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end,c_score,c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

@torch.no_grad()
def multiple_uncertainty(vid_name, inputs):
    score_act, score_bkg, feat_act, feat_bkg, features, cas_softmax = inputs['score_act'], inputs['score_bkg'], inputs['feat_act'], inputs['feat_bkg'], inputs['features'], inputs['cas']
    num_segments = cas_softmax.shape[-2]
    act_thresh_cas = np.arange(0.0, 0.25, 0.025)
    act_thresh_magnitudes = np.arange(0.4, 0.625, 0.025)
    class_thresh = 0.2
    #cas_softmax [batch,neg_seg,ctg_prob]
    feat_magnitudes_act = torch.mean(torch.norm(feat_act, dim=2), dim=1)
    feat_magnitudes_bkg = torch.mean(torch.norm(feat_bkg, dim=2), dim=1)

    # label_np = _label.cpu().data.numpy()
    score_np = score_act[0].cpu().data.numpy()

    pred_np = np.zeros_like(score_np)
    pred_np[np.where(score_np < class_thresh)] = 0
    pred_np[np.where(score_np >= class_thresh)] = 1
    #预测发生的类

    # #预测正确的类，包括负类和正类
    # correct_pred = np.sum(label_np == pred_np, axis=1)

    # #整个视频类预测正确的，加项为1或0
    # num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
    # num_total += correct_pred.shape[0]

    feat_magnitudes = torch.norm(features, p=2, dim=2)

    feat_magnitudes = utils.minmax_norm(feat_magnitudes, max_val=feat_magnitudes_act, min_val=feat_magnitudes_bkg)
    feat_magnitudes = feat_magnitudes.repeat((args.num_class, 1, 1)).permute(1, 2, 0)

    cas = utils.minmax_norm(cas_softmax * feat_magnitudes)

    pred = np.where(score_np >= class_thresh)[0]

    if len(pred) == 0:
        pred = np.array([np.argmax(score_np)])

    cas_pred = cas[0].cpu().numpy()[:, pred] #预测类的分值
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1)) #[num_segments,>threshold,1]
    cas_pred = utils.upgrade_resolution(cas_pred, args.scale)

    proposal_dict = {}
    feat_magnitudes_np = feat_magnitudes[0].cpu().data.numpy()[:, pred]
    feat_magnitudes_np = np.reshape(feat_magnitudes_np, (num_segments, -1, 1))
    feat_magnitudes_np = utils.upgrade_resolution(feat_magnitudes_np, args.scale)
    # print(vid_num_seg,num_segments)

    # for i in range(len(act_thresh_cas)):
    #     cas_temp = cas_pred.copy()

    #     zero_location = np.where(cas_temp[:, :, 0] < act_thresh_cas[i])
    #     cas_temp[zero_location] = 0
        
    #     seg_list = []
    #     for c in range(len(pred)):
    #         pos = np.where(cas_temp[:, c, 0] > 0)
    #         seg_list.append(pos)
    #     proposals = utils.get_proposal_oic_2(seg_list, cas_temp, score_np, pred, args.scale, num_segments, args.feature_fps, num_segments)

    #     for i in range(len(proposals)):
    #         class_id = proposals[i][0][0]

    #         if class_id not in proposal_dict.keys():
    #             proposal_dict[class_id] = []

    #         proposal_dict[class_id] += proposals[i]

    for i in range(len(act_thresh_magnitudes)):
        cas_temp = cas_pred.copy()

        feat_magnitudes_np_temp = feat_magnitudes_np.copy()

        zero_location = np.where(feat_magnitudes_np_temp[:, :, 0] < act_thresh_magnitudes[i])
        feat_magnitudes_np_temp[zero_location] = 0
        
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(feat_magnitudes_np_temp[:, c, 0] > 0)
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list, cas_temp, score_np, pred, args.scale, \
                        num_segments, args.feature_fps, num_segments)

        for i in range(len(proposals)):
            class_id = proposals[i][0][0]

            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []

            proposal_dict[class_id] += proposals[i]
    
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))
    # self.final_res["results"][vid_name[0]] = utils.result2json(
        # final_proposals, class_dict)

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    #[c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end,c_score,c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction
   

def normal_threshold(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp)
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                segment_predict.append(
                    [s[j], e[j], np.max(seg)+0.7*c_s[c],c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

def normal_threshold_ant(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp) if 'Thumos' in args.dataset_name else np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thres
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                segment_predict.append(
                    [s[j], e[j], np.max(seg)+0.7*c_s[c],c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction


def normal_threshold_IOC(vid_name,prediction):


    # _data, _label, _, vid_name, vid_num_seg = inputs
    # attn,att_logits,feat_embed, bag_logit, prediction = outputs
    # attn,feat_embed, bag_logit, prediction = outputs
    prediction = prediction[0].cpu().numpy()
    # process the predictions such that classes having greater than a certain threshold are detected only
    prediction_mod = []
    prediction, c_s = _get_vid_score(prediction)
    if args is None:
        thres = 0.5
    else:
        thres = 1 - args.thres

    softmaxed_c_s=np.exp(c_s)/np.sum(np.exp(c_s))
    c_set = []
    for c in range(c_s.shape[0]):
        if np.max(softmaxed_c_s)>=0.15:
            if softmaxed_c_s[c]<0.15:
                continue
            else:
                c_set.append(c)
        else:
            if c != np.argsort(c_s,axis=-1)[-1]:
                continue
            else:
                c_set.append(c)
    # for c in np.argsort(c_s,axis=-1):
    #     if softmaxed_c_s[c]>=0.15:
    #         c_set.append(c)
    # if len(c_set)==0:
    #     c_set.append(np.argsort(c_s,axis=-1)[-1])
    
    segment_predict = []
    for c in c_set:
        # prediction[:,c] = __vector_minmax_norm(prediction[:,c])
        tmp=prediction[:,c] if 'Thumos' in args.dataset_name else smooth(prediction[:,c])
        threshold=np.mean(tmp) if 'Thumos' in args.dataset_name else np.max(tmp) - (np.max(tmp) - np.min(tmp)) * thres
        vid_pred = np.concatenate(
            [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)], axis=0
        )
        vid_pred_diff = [
            vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
        ]
        s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]  #the start point set
        e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]  #the end point set
        for j in range(len(s)):
            if e[j] - s[j] >= 1:
                seg=tmp[s[j] : e[j]]
                inner_score = np.mean(seg)

                len_proposal = len(seg)

                outer_s = max(0, int(s[j] - 0.25 * len_proposal))
                outer_e = min(int(prediction.shape[0] - 1), int(e[j]+ 0.25 * len_proposal))

                outer_temp_list = list(range(outer_s, s[j])) + list(range(int(e[j] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(tmp[outer_temp_list])
                
                c_score = inner_score - outer_score + 0.7*c_s[c]

                segment_predict.append(
                    [s[j], e[j], c_score,c]
                    # [c,np.max(seg)+0.7*c_s[c],s[j], e[j]]
                )
    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())
    
   
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction
