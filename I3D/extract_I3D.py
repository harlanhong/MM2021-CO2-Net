import os
import sys
sys.path.append('../../../')
import argparse
from UCFCrime.SpatioTemporal_Detection.I3D_STD import *
import h5py
import numpy as np
import cv2
from opencv_videovision import transforms
from tqdm import tqdm
from UCFCrime.SpatioTemporal_Detection import constant
from UCFCrime.SpatioTemporal_Detection.CAM import softmaxed_weight_mlp,get_CAM
from UCFCrime.SpatioTemporal_Detection.Grad_CAM import GradCAM
import torch
from apex import amp

def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('--MODEL',type=str,default='I3D')
    parser.add_argument('--train',type=str,default='TD')

    parser.add_argument('--label_dir',type=str,default='jiachang/Pseudo_Labels_MultiViews/UCF_PLs_I3D_SGA_iter_1_AC/')
    parser.add_argument('--merge_h5_path',type=str,default='jiachang/UCF_I3D_SGA_iter_1_AC.h5')
    parser.add_argument('--feature_dir',type=str,default='jiachang/UCF_I3D_SGA_iter_1_AC/')
    parser.add_argument('--frames_path',type=str,default='jiachang/UCFCrime-Frames-16.h5')
    parser.add_argument('--test_frames_path',type=str,default='jiachang/UCFCrime-Frames-test-16.h5')

    parser.add_argument('--test',dest='test_extract',action='store_true')
    parser.set_defaults(test_extract=False)

    parser.add_argument('--logits_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Final_Vis_Logits/UCF_I3D_SGA/')
    parser.add_argument('--atten_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Final_Vis_Atten/UCF_I3D_SGA/')
    parser.add_argument('--merge_att_h5_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Final_Vis_Atten/UCF_I3D_SGA/att_maps.h5')
    parser.add_argument('--pretrained_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Final_Vis_Models/UCF_I3D_SGA_epoch_270_AUC_0.82298.pth')
    parser.add_argument('--CAM_path',type=str,default='/jiachang/Weakly_Supervised_VAD/Final_Vis_CAMs/UCF_I3D_SGA/')

    parser.add_argument('--machine',type=str,default='data2')
    parser.add_argument('--gpus',type=str,default='0,1,2,3')

    parser.add_argument('--ten_crop',dest='use_ten_crop',action='store_true')
    parser.add_argument('--dropout_rate',type=float,default=0)
    parser.add_argument('--expand_k',type=int,default=8)
    parser.add_argument('--gap',type=int,default=1)
    parser.add_argument('--num_frames',type=int,default=16)

    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--split_len',type=int,default=50)
    parser.add_argument('--gpu',type=int,default=0)

    parser.add_argument('--merge',dest='merge_files',action='store_true')

    parser.add_argument('--origin',dest='use_origin',action='store_true')
    parser.set_defaults(use_origin=False)
    parser.set_defaults(merge_files=False)
    parser.set_defaults(use_ten_crop=False)
    args=parser.parse_args()

    args.machine='/'+args.machine
    args.label_dir=args.machine+'/'+args.label_dir
    args.feature_dir=args.machine+'/'+args.feature_dir
    args.pretrained_path=args.machine+'/'+args.pretrained_path
    args.frames_path=args.machine+'/'+args.frames_path
    args.test_frames_path=args.machine+'/'+args.test_frames_path

    args.merge_h5_path=args.machine+'/'+args.merge_h5_path

    # os.environ['CUDA_VISIBLE_DEVICES'] =args.gpus
    # args.gpus=[int(i)for i in args.gpus.split(',')]
    args.logits_path=args.machine+'/'+args.logits_path
    args.atten_path=args.machine+'/'+args.atten_path
    args.merge_att_h5_path=args.machine+'/'+args.merge_att_h5_path
    args.CAM_path=args.machine+'/'+args.CAM_path
    return args


from torch.utils.data import Dataset,DataLoader
class Extract_Dataset(Dataset):
    def __init__(self,keys,h5_file,ten_crop=False,num_frames=16):
        self.h5_file=h5_file
        self.ten_crop = ten_crop
        self.num_frames=num_frames

        self.mean=[128,128,128]
        self.std=[128,128,128]

        self.transforms = transforms.Compose([transforms.Resize([240,320]),
                                              # transforms.CenterCrop([112,112]),
                                              transforms.ClipToTensor(div_255=False),
                                              transforms.Normalize(self.mean,self.std)])
        self.ten_crop_aug=transforms.Compose([transforms.Resize([256, 340]),
                                           transforms.ClipToTensor(div_255=False),
                                           transforms.Normalize(mean=self.mean, std=self.std),
                                           transforms.TenCropTensor(224)])
        self.keys=keys
        self.segment_len = h5py.File(self.h5_file, 'r')[keys[0]][:].__len__()


    # def set_keys(self, key):
    #     self.key = key
    #     keys = list(h5py.File(self.h5_file, 'r').keys())
    #     self.segment_len = h5py.File(self.h5_file, 'r')[keys[0]][:].__len__()
    #     self.keys = []
    #     for k in keys:
    #         if key == k.split('-')[0]:
    #             self.keys.append(k)

    def __len__(self):
        return len(self.keys)*self.segment_len//self.num_frames

    def decode_imgs(self,i):
        # assert len(frames)==self.num_frames,'Frames is not enough! with only {}'.format(len(frames))

        # frames=[]
        with h5py.File(self.h5_file, 'r',swmr=True) as h5:
            # for j in range(self.num_frames):
            #     seg_p = (i * self.num_frames+j) // self.segment_len
            #     seg_id = (i * self.num_frames+j) % self.segment_len
            frames=h5[self.keys[i]][:]
        frames=[(cv2.cvtColor(cv2.imdecode(np.frombuffer(frame,np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))for frame in frames]
        # frames=[]
        # for frame in encoded_frames:
        # frames=[cv2.imdecode(np.frombuffer(frame,np.uint8), cv2.IMREAD_COLOR) for frame in encoded_frames]
        return frames

    def __getitem__(self, i):
        # with h5py.File(self.h5_file,'r') as h5:
        frames=self.decode_imgs(i)
        if self.ten_crop:
            frames=self.ten_crop_aug(frames)
            frames = torch.stack(frames, dim=0)
        else:
            frames = self.transforms(frames)

        return frames,self.keys[i]

def extract_one(h5_path,vid_path,model):
    vc=cv2.VideoCapture(vid_path)
    vc_len=int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    feats=[]
    for i in tqdm(range(vc_len//16)):
        frames=[]
        for j in range(16):
            _,frame=vc.read()
            frame=frame[...,::-1]
            # frame=cv2.resize(frame,(240,128))
            # frame=frame[8:-8,29:-30]
            frame=frame.astype(np.float32)
            frame=(frame-128.)/128.
            # frame=frame-[90.25,97.66,101.41]
            frames.append(frame)
        frames=np.array(frames).transpose([3,0,1,2])
        frames=torch.from_numpy(frames).cuda().float().unsqueeze(0)
        feat = model(frames)[1]
        feat = feat.view([-1, 1024, feat.shape[-3], feat.shape[-2], feat.shape[-1]]).mean(-1).mean(-1).mean(
            -1).detach().cpu().numpy().squeeze()
        feats.append(feat)

    feats=np.array(feats)
    with h5py.File(h5_path,'a') as h5:
        h5.create_dataset(name=vid_path.split('/')[-1].split('.')[0]+'.npy',data=feats)
    print('finished!')


class Extract_Dataset_Vis(Dataset):
    def __init__(self,keys,h5_file,ten_crop=False,num_frames=16):
        self.h5_file=h5_file
        self.ten_crop = ten_crop
        self.num_frames=num_frames

        self.mean=[128,128,128]
        self.std=[128,128,128]

        self.transforms = transforms.Compose([transforms.Resize([240,320]),
                                              # transforms.CenterCrop([112,112]),
                                              transforms.ClipToTensor(div_255=False),
                                              transforms.Normalize(self.mean,self.std)])
        self.ten_crop_aug=transforms.Compose([transforms.Resize([256, 340]),
                                           transforms.ClipToTensor(div_255=False),
                                           transforms.Normalize(mean=self.mean, std=self.std),
                                           transforms.TenCropTensor(224)])


        # self.set_keys(key)
        self.segment_len = h5py.File(self.h5_file, 'r')[keys[0]][:].__len__()
        self.keys=keys

    # def set_keys(self, key):
    #     self.key = key
    #     keys = list(h5py.File(self.h5_file, 'r').keys())
    #     self.keys = []
    #     for k in keys:
    #         if key == k.split('-')[0]:
    #             self.keys.append(k)

    def __len__(self):
        return len(self.keys)*self.segment_len//self.num_frames

    def decode_imgs(self,i):
        # assert len(frames)==self.num_frames,'Frames is not enough! with only {}'.format(len(frames))
        # seg_p=i*self.num_frames//self.segment_len
        # seg_id=i*self.num_frames%self.segment_len
        # frames=[]
        with h5py.File(self.h5_file, 'r',swmr=True) as h5:
            frames=h5[self.keys[i]][:]
            # for j in range(self.num_frames):
                # frames.append(cv2.imdecode(np.frombuffer(h5[self.key + '-{0:06d}'.format(seg_p)][:][seg_id+j],np.uint8), cv2.IMREAD_COLOR))
        frames=[cv2.cvtColor(cv2.imdecode(np.frombuffer(frame,np.uint8), cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB) for frame in frames]
        # frames=[]
        # for frame in encoded_frames:
        # frames=[cv2.imdecode(np.frombuffer(frame,np.uint8), cv2.IMREAD_COLOR) for frame in encoded_frames]
        return frames

    def __getitem__(self, i):
        # with h5py.File(self.h5_file,'r') as h5:
        frames=self.decode_imgs(i)
        if self.ten_crop:
            frames=self.ten_crop_aug(frames)
            frames = torch.stack(frames, dim=0)
        else:
            frames = self.transforms(frames)

        return frames,self.keys[i]

def extract_process(args,keys,model):

        if args.test_extract:
            dataset = Extract_Dataset(keys, args.test_frames_path, args.use_ten_crop,
                                      args.num_frames)
        else:
            dataset = Extract_Dataset(keys, args.frames_path, args.use_ten_crop,
                                      args.num_frames)

        dataloader = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=False, num_workers=5, pin_memory=True)
        feat_dict = {}
        score_dict = {}

        for j,(frames,_keys) in enumerate(tqdm(dataloader)):
            if args.use_ten_crop:
                frames = frames.contiguous().view([-1, 3, args.num_frames, 224, 224]).cuda().float()
            else:
                frames = frames.contiguous().view([-1, 3, args.num_frames, 240, 320]).cuda().float()

            with torch.no_grad():
                if not args.use_origin:
                    score, feat = model(frames)[:2]
                else:
                    feat=model(frames)[0]
            if args.use_ten_crop:
                feat=feat.view([-1,10,1024,feat.shape[-3],feat.shape[-2],feat.shape[-1]]).mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                if not args.use_origin:
                    score=score.view([-1,10,2]).detach().cpu().numpy()
            else:
                feat=feat.view([-1,1024,feat.shape[-3],feat.shape[-2],feat.shape[-1]]).mean(-1).mean(-1).mean(-1).detach().cpu().numpy()
                if not args.use_origin:
                    score=score.view([-1,2]).detach().cpu().numpy()
            for j, key in enumerate(_keys):
                if not args.use_origin:
                    score_dict[key] = score[j]
                feat_dict[key] = feat[j]
        if not args.test_extract:
            np.save(os.path.join(args.feature_dir, 'train-{}.npy'.format(args.gpu)), feat_dict)
            if not args.use_origin:
                np.save(os.path.join(args.label_dir, 'split/train-{}.npy'.format(args.gpu)), score_dict)
        else:
            np.save(os.path.join(args.feature_dir, 'test-{}.npy'.format(args.gpu)), feat_dict)
            if not args.use_origin:
                np.save(os.path.join(args.label_dir, 'split/test-{}.npy'.format(args.gpu)), score_dict)



def softmaxed_weight_att(weight,expand_k=8):
    weight=weight.squeeze()
    end_weights=torch.cat([torch.tensor([[1./expand_k,0]]).repeat([expand_k,1]),torch.tensor([[0.,1./expand_k]]).repeat([expand_k,1])],dim=0).to(weight.device)
    # import pdb
    # pdb.set_trace()
    softmaxed_weights=torch.mm(end_weights.t(),torch.softmax(weight,dim=-1))
    return softmaxed_weights

def extract_process_vis(args,keys,model):
    dataset = Extract_Dataset_Vis(keys, args.test_frames_path, args.use_ten_crop, args.num_frames)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=5, pin_memory=True)
    # score_dict={}
    # att_scores_dict={}
    # atten_maps_dict={}
    # cam_maps_cls_dict={}
    # cam_maps_att_dict={}
    # GC_cls_dict={}
    # GC_att_dict={}
    #
    # with torch.no_grad():
    #     softmaxed_weights_cls=softmaxed_weight_mlp(model.Regressor)#,softmaxed_weight_mlp(model.module.fc6))
    #     softmaxed_weights_att=softmaxed_weight_att(model.Conv_Atten.Att_2.weight)
    # gradcam_cls = GradCAM(model.Regressor, grad_pp=False,act=False)
    # gradcam_att=GradCAM(model.Conv_Atten.Att_2,grad_pp=False,att=True,expand_k=args.expand_k,act=False)

    data_list=list(dataloader)
    import time
    begin=time.time()
    for j,(frames,_keys) in tqdm(enumerate(data_list)):
        frames = frames.contiguous().view([-1, 3, args.num_frames, 240, 320]).cuda().float()

        with torch.no_grad():
            logits, feat_maps,atten_logits,atten_maps,att_feat_maps = model(frames,act=True,extract=True)
    end=time.time()
    gap=end-begin
    print('take times for {}'.format(gap))
    #         logits=logits.detach().cpu().numpy()
    #         atten_logits=atten_logits.detach().cpu().numpy()
    #         atten_maps=atten_maps.detach().cpu().numpy()
    #
    #     for j,key in enumerate(_keys):
    #         score_dict[key]=logits[j]
    #         att_scores_dict[key]=atten_logits[j]
    #         atten_maps_dict[key]=atten_maps[j]
    #         cam_maps_cls_dict[key]=get_CAM(feat_maps[j].unsqueeze(0), softmaxed_weights_cls)[1].squeeze().detach().cpu().numpy()
    #         cam_maps_att_dict[key]=get_CAM(att_feat_maps[j].unsqueeze(0),softmaxed_weights_att)[1].squeeze().detach().cpu().numpy()
    #         GC_att_dict[key]=gradcam_att(att_feat_maps[j].unsqueeze(0))
    #         GC_cls_dict[key]=gradcam_cls(feat_maps[j].unsqueeze(0))
    #
    # np.save(os.path.join(args.logits_path,'test-{}.npy'.format(args.gpu)),score_dict)
    # np.save(os.path.join(args.logits_path,'test-att-{}.npy'.format(args.gpu)),att_scores_dict)
    # np.save(os.path.join(args.atten_path,'test-{}.npy'.format(args.gpu)),atten_maps_dict)
    # np.save(os.path.join(args.CAM_path,'CAM-cls-{}.npy').format(args.gpu),cam_maps_cls_dict)
    # np.save(os.path.join(args.CAM_path,'CAM-att-{}.npy').format(args.gpu),cam_maps_att_dict)
    # np.save(os.path.join(args.CAM_path,'GC-cls-{}.npy').format(args.gpu),GC_cls_dict)
    # np.save(os.path.join(args.CAM_path,'GC-att-{}.npy').format(args.gpu),GC_att_dict)
    print('finished!')

    # print('finished! with time of ')

    # for i in tqdm(range(len(keys))):
    #     key=keys[i]
    #     if args.test_extract:
    #         dataset = Extract_Dataset(keys[i], args.test_frames_path, args.use_ten_crop,
    #                                   args.num_frames)
    #     else:
    #         dataset = Extract_Dataset(keys[i], args.frames_path, args.use_ten_crop,
    #                                   args.num_frames)
    #     if os.path.exists(os.path.join(args.label_dir, key + '-{0:06d}'.format(len(dataset)-1) + '.npy')):
    #         continue
    #     dataloader = DataLoader(dataset, batch_size=5, shuffle=False, drop_last=False, num_workers=5, pin_memory=True)
    #
    #     # data_iter=dataloader.__iter__()
    #     # next_batch=data_iter.__next__()
    #     # next_batch=next_batch.cuda(non_blocking=True).float()
    #     # next_batch=next_batch.contiguous().view([-1,3,args.num_frames,112,112])
    #     # for j in tqdm(range(len(dataloader))):
    #     #     batch = next_batch
    #     #     if j + 1 != len(dataloader):
    #     #         next_batch = data_iter.__next__()
    #     #         # next_batch = next_batch.cuda(non_blocking=True).float().contiguous().permute([0, 1, 5, 2, 3, 4]).contiguous().view(
    #     #         #     [-1, 3, args.num_frames, 224, 224])
    #     #         next_batch = next_batch.cuda(non_blocking=True).float().contiguous().view(
    #     #             [-1, 3, args.num_frames, 112, 112])
    #     with torch.no_grad():
    #         for j,frames in enumerate(dataloader):
    #             frames=frames.view([-1, 3, args.num_frames, 112, 112])
    #             feat=model(frames)
    #             feat=feat.mean(dim=-1).mean(dim=-1).mean(dim=-1).detach().cpu().numpy()
    #             np.save(os.path.join(args.feature_dir,key+'-{0:06d}'.format(j)+ '.npy'),feat)
    #     print(key)

def merge_vis(args):
    score_dict={}
    att_score_dict={}
    att_maps_dict={}
    CAM_cls_dict={}
    CAM_att_dict={}
    GC_att_dict={}
    GC_cls_dict={}

    for i,path in tqdm(enumerate(os.listdir(args.atten_path))):
        tmp_att_map_dict=np.load(os.path.join(args.atten_path,path),allow_pickle=True).tolist()
        tmp_score_dict=np.load(os.path.join(args.logits_path,'test-{}.npy'.format(i)),allow_pickle=True).tolist()
        tmp_att_score_dict=np.load(os.path.join(args.logits_path,'test-att-{}.npy'.format(i)),allow_pickle=True).tolist()
        tmp_CAM_cls_dict=np.load(os.path.join(args.CAM_path,'CAM-cls-{}.npy'.format(i)),allow_pickle=True).tolist()
        tmp_CAM_att_dict=np.load(os.path.join(args.CAM_path,'CAM-att-{}.npy'.format(i)),allow_pickle=True).tolist()
        tmp_GC_att_dict=np.load(os.path.join(args.CAM_path,'GC-att-{}.npy'.format(i)),allow_pickle=True).tolist()
        tmp_GC_cls_dict=np.load(os.path.join(args.CAM_path,'GC-cls-{}.npy'.format(i)),allow_pickle=True).tolist()

        if i==0:
            att_maps_dict = tmp_att_map_dict.copy()
            score_dict = tmp_score_dict.copy()
            att_score_dict = tmp_att_score_dict.copy()
            CAM_cls_dict = tmp_CAM_cls_dict.copy()
            CAM_att_dict=tmp_CAM_att_dict.copy()
            GC_cls_dict = tmp_GC_cls_dict.copy()
            GC_att_dict=tmp_GC_att_dict.copy()
        else:
            att_maps_dict = dict(att_maps_dict, **tmp_att_map_dict)
            score_dict = dict(score_dict, **tmp_score_dict)
            att_score_dict = dict(att_score_dict, **tmp_att_score_dict)
            CAM_cls_dict = dict(CAM_cls_dict,**tmp_CAM_cls_dict)
            CAM_att_dict=dict(CAM_att_dict,**tmp_CAM_att_dict)
            GC_cls_dict = dict(GC_cls_dict,**tmp_GC_cls_dict)
            GC_att_dict=dict(GC_att_dict,**tmp_GC_att_dict)

    att_keys = list(att_maps_dict.keys())
    vid_key_dict = {}
    out_score_dict={}
    out_att_score_dict={}
    out_att_dict={}
    out_CAM_att_map_dict={}
    out_CAM_cls_map_dict={}
    out_GC_att_map_dict={}
    out_GC_cls_map_dict={}
    for key in att_keys:
        k = key.split('-')[0]
        if k not in vid_key_dict.keys():
            vid_key_dict[k] = 1
        else:
            vid_key_dict[k] += 1
    h5_keys=list(h5py.File(args.frames_path,'r').keys())
    h5_keys=list(set([key.split('-')[0] for key in h5_keys]))
    vid_keys=list(vid_key_dict.keys())
    # for key in h5_keys:
    #     if key not in vid_keys:
    #         print(key)

    # import pdb
    # pdb.set_trace()

    # with h5py.File(args.merge_att_h5_path,'a') as h5:
    for key in tqdm(vid_keys):
        att_maps=[]
        scores=[]
        att_scores=[]
        CAMs_cls=[]
        CAMs_att=[]
        GC_cls=[]
        GC_att=[]

        for i in tqdm(range(vid_key_dict[key])):
            att_map = att_maps_dict[key + '-{0:06d}'.format(i)]
            att_maps.append(att_map)
            score=score_dict[key+'-{0:06d}'.format(i)]
            scores.append(score)
            att_score=score_dict[key+'-{0:06d}'.format(i)]
            att_scores.append(att_score)
            CAMs_cls.append(CAM_cls_dict[key+'-{0:06d}'.format(i)])
            CAMs_att.append(CAM_att_dict[key+'-{0:06d}'.format(i)])
            GC_cls.append(GC_cls_dict[key+'-{0:06d}'.format(i)])
            GC_att.append(GC_att_dict[key+'-{0:06d}'.format(i)])
        out_att_dict[key+'.npy']=np.array(att_maps)
        out_score_dict[key+'.npy']=np.array(scores)
        out_att_score_dict[key+'.npy']=np.array(att_scores)
        out_CAM_cls_map_dict[key+'.npy']=np.array(CAMs_cls)
        out_CAM_att_map_dict[key+'.npy']=np.array(CAMs_att)
        out_GC_cls_map_dict[key+'.npy']=np.array(GC_cls)
        out_GC_att_map_dict[key+'.npy']=np.array(GC_att)
    np.save(os.path.join(args.logits_path,'logits.npy'),out_score_dict)
    np.save(os.path.join(args.logits_path,'att_logits.npy'),out_att_score_dict)
    np.save(os.path.join(args.atten_path,'att_maps.npy'),out_att_dict)
    np.save(os.path.join(args.CAM_path,'CAM_att.npy'),out_CAM_att_map_dict)
    np.save(os.path.join(args.CAM_path,'CAM_cls.npy'),out_CAM_cls_map_dict)
    np.save(os.path.join(args.CAM_path,'GC_att.npy'),out_GC_att_map_dict)
    np.save(os.path.join(args.CAM_path,'GC_cls.npy'),out_GC_cls_map_dict)
    print('finished!')

def get_model(args):
    if args.train =='TD':
        model=I3D_TD_Module(args.dropout_rate,
                                    freeze_backbone=True,freeze_bn=True).cuda()
    elif args.train=='SG':
        model=I3D_SG_Module(args.dropout_rate,args.expand_k,
                                    freeze_backbone=True,freeze_bn=True
                                     ).cuda()
    elif args.train=='SGA':
        model=I3D_SGA_Module(args.dropout_rate,args.expand_k,freeze_backbone=True,freeze_bn=True,freeze_bn_statics=True).cuda()
    stored_dict=torch.load(args.pretrained_path)['model']
    new_dict={k[7:]:v for k,v in stored_dict.items()}
    model.load_state_dict(new_dict)

    return model

def get_origin_model(args):
    model=I3D().cuda()
    CFG=constant._C

    pretrained_dict=torch.load(args.machine+CFG.I3D_MODEL_PATH)
    state_dict=model.state_dict()
    new_dict={k:v for k,v in pretrained_dict.items() if k in state_dict.keys()}
    state_dict.update(new_dict)
    model.load_state_dict(state_dict)
    return model

def mkdir(dir):
    try:os.mkdir(dir)
    except:pass

def merge(args):
    # mkdir(os.path.join(args.feature_dir,'merged'))
    # score_paths=os.listdir(args.label_dir)
    feat_dict={}
    score_dict={}

    for i,path in tqdm(enumerate(os.listdir(args.feature_dir))):
        if 'merged' != path:
            tmp_feat_dict=np.load(os.path.join(args.feature_dir,path),allow_pickle=True).tolist()
            if i==0:
                feat_dict=tmp_feat_dict.copy()
            else:
                feat_dict=dict(feat_dict,**tmp_feat_dict)
            if not args.use_origin:
                tmp_score_dict=np.load(os.path.join(args.label_dir,'split',path),allow_pickle=True).tolist()
                if i==0:
                    score_dict=tmp_score_dict.copy()
                else:
                    score_dict=dict(score_dict,**tmp_score_dict)#score_dict.update(tmp_score_dict)

    vid_keys = sorted(list(set([key.split('-')[0] for key in list(feat_dict.keys())])))
    keys = list(feat_dict.keys())

    vid_key_dict = {}
    out_score_dict={}
    for key in keys:
        k = key.split('-')[0]
        if k not in vid_key_dict.keys():
            vid_key_dict[k] = 1
        else:
            vid_key_dict[k] += 1


    h5_keys=list(h5py.File(args.frames_path,'r').keys())
    h5_keys=list(set([key.split('-')[0] for key in h5_keys]))
    vid_keys=list(vid_key_dict.keys())
    for key in h5_keys:
        if key not in vid_keys:
            print(key)
    # import pdb
    # pdb.set_trace()

    with h5py.File(args.merge_h5_path,'w') as h5:
        for key in vid_keys:
            features=[]
            scores=[]
            for i in tqdm(range(vid_key_dict[key])):
                feature = feat_dict[key + '-{0:06d}'.format(i)]
                features.append(feature)
                if not args.use_origin:
                    score=score_dict[key+'-{0:06d}'.format(i)]
                    scores.append(score)

            h5.create_dataset(key+'.npy',data=np.array(features),chunks=True)
            if not args.use_origin:
                out_score_dict[key+'.npy']=np.array(scores)
    if not args.use_origin:
        np.save(os.path.join(args.label_dir,'pseudo_labels.npy'),out_score_dict)
    print('finished!')


# def merge(args):
#     key_dict = {}
#     score_dict={}
#
#     for path in os.listdir(args.feature_dir):
#         # if 'merged' != path:
#         key = path.split('-')[0]
#         if key not in key_dict.keys():
#             key_dict[key] = 1
#         else:
#             key_dict[key] += 1
#         # mkdir(os.path.join(args.feature_dir, 'merged'))
#     with h5py.File(args.merge_h5_path, 'w') as h5:
#         for key in tqdm(key_dict.keys()):
#             features = []
#             scores=[]
#             for i in tqdm(range(key_dict[key])):
#                 feature = np.load(os.path.join(args.feature_dir, key + '-{0:06d}.npy'.format(i)))
#                 features.extend(feature.tolist())
#                 if not args.use_origin:
#                     score=np.load(os.path.join(args.label_dir,'split',key+'-{0:06d}.npy'.format(i)))
#                     scores.extend(score.tolist())
#
#             h5.create_dataset(key+'.npy',data=np.array(features),chunks=True)
#             if not args.use_origin:
#                 score_dict[key+'.npy']=np.array(scores)
#     if not args.use_origin:
#         np.save(os.path.join(args.label_dir,'pseudo_labels.npy'),score_dict)
#     print('finished!')

if __name__=='__main__':
    args=parse_args()
    model=get_model(args).cuda().eval()
    extract_one('/data2/jiachang/UCF_I3D_SGA_iter_1_AC.h5','/mnt/sdd/jiachang/Anomaly-Videos/Abuse/Abuse048_x264.mp4',model)
    # if not args.merge_files:
    #     if not args.test_extract:
    #         with h5py.File(args.frames_path, 'r') as h5:
    #             # keys = sorted(list(set([key.split('-')[0] for key in list(h5.keys())])))
    #             keys=list(h5.keys())
    #
    #     else:
    #         with h5py.File(args.test_frames_path, 'r') as h5:
    #             # keys = sorted(list(set([key.split('-')[0] for key in list(h5.keys())])))
    #             keys = list(h5.keys())
    #     os.environ['CUDA_VISIBLE_DEVICES'] =str(args.gpu)
    #     if not args.use_origin:
    #         model=get_model(args).cuda()
    #     else:
    #         model=get_origin_model(args).cuda()
    #     model=model.eval()
    #     import random
    #     random.seed(0)
    #     random.shuffle(keys)
    #     index_len=len(keys)//args.split_len+1
    #     if args.index+1==index_len:
    #         keys=keys[args.index*args.split_len:]
    #     else:
    #         keys=keys[args.index*args.split_len:(args.index+1)*args.split_len]
    #     # for key in tqdm(keys):
    #     extract_process(args,keys,model)
    # else:
    #     merge(args)

# if __name__ == '__main__':
#     args = parse_args()
#     mkdir(args.logits_path)
#     mkdir(args.atten_path)
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#     import random
#     random.seed(0)
#
#     if not args.merge_files:
#         model=get_model(args).cuda().eval()
#         with h5py.File(args.test_frames_path, 'r') as h5:
#             keys=list(h5.keys())
#         random.shuffle(keys)
#         index_len = len(keys) // args.split_len + 1
#         if args.index + 1 == index_len:
#             keys = keys[args.index * args.split_len:]
#         else:
#             keys = keys[args.index * args.split_len:(args.index + 1) * args.split_len]
#         extract_process_vis(args, keys, model)
#     else:
#         merge_vis(args)