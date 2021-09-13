from __future__ import print_function
import argparse
import os
import torch
# from model import Model
import model
import multiprocessing as mp
import wsad_dataset
import pdb
import random
# from video_dataset import Dataset,ANTDataset
from test import test
from train import train
from tensorboard_logger import Logger
# from utils import Recorder as Logger
import options
import numpy as np
from torch.optim import lr_scheduler
from tqdm import tqdm
import shutil
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

import torch.optim as optim

if __name__ == '__main__':
   pool = mp.Pool(5)

   args = options.parser.parse_args()
   # seed = random.randint(1,10000)
   seed=args.seed
   print('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
   setup_seed(seed)
   # torch.manual_seed(args.seed)
   device = torch.device("cuda")
   dataset = getattr(wsad_dataset,args.dataset)(args)
   if 'Thumos' in args.dataset_name:
      max_map=[0]*9
   else:
      max_map=[0]*10
   if not os.path.exists('./ckpt/'):
      os.makedirs('./ckpt/')
   if not os.path.exists('./logs/' + args.model_name):
      os.makedirs('./logs/' + args.model_name)
   if os.path.exists('./logs/' + args.model_name):
      shutil.rmtree('./logs/' + args.model_name)
   logger = Logger('./logs/' + args.model_name)
   print(args)
   # model = Model(dataset.feature_size, dataset.num_class).to(device)
   model = getattr(model,args.use_model)(dataset.feature_size, dataset.num_class,opt=args).to(device)

   if args.pretrained_ckpt is not None:
      model.load_state_dict(torch.load(args.pretrained_ckpt))

   optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   # optimizer = optim.SGD(model.parameters(), lr=args.lr,
                  #   momentum=args.momentum, weight_decay=args.weight_decay)
   # scheduler = lr_scheduler.StepLR(optimizer,step_size = args.lr_decay,gamma = 0.5)

   total_loss = 0
   lrs = [args.lr, args.lr/5, args.lr/5/5]
   print(model)
   for itr in tqdm(range(args.max_iter)):
      loss = train(itr, dataset, args, model, optimizer, logger, device)
      total_loss+=loss
      if itr % args.interval == 0 and not itr == 0:
         print('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval))
         total_loss = 0
         torch.save(model.state_dict(), './ckpt/last_' + args.model_name + '.pkl')
         iou,dmap = test(itr, dataset, args, model, logger, device,pool)
         if 'Thumos' in args.dataset_name:
            cond=sum(dmap[:7])>sum(max_map[:7])
         else:
            cond=np.mean(dmap)>np.mean(max_map)
         if cond:
            torch.save(model.state_dict(), './ckpt/best_' + args.model_name + '.pkl')
            max_map = dmap


         print('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
         max_map = np.array(max_map)
         print('mAP Avg 0.1-0.5: {}, mAP Avg 0.1-0.7: {}, mAP Avg ALL: {}'.format(np.mean(max_map[:5])*100,np.mean(max_map[:7])*100,np.mean(max_map)*100))
         print("------------------pid: {}--------------------".format(os.getpid()))

    
