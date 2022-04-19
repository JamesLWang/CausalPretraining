import torchvision
from utils import *
import numpy as np

import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os, socket, random

from models.preactresnet import PreActResNet18_encoder, VAE_Small, FDC_deep_preact

from torchvision.utils import save_image

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

from models.resnet import resnet18, FCClassifier, resnet50, FDC, FC2Classifier, FDC5
from dataloader.multidomain_loader import MultiDomainLoader, DomainTest, RandomData
    
domain_model = torch.load("/proj/vondrick3/james/AudioDefense_/Control/results/pacs_cat_/2022-04-16_01:05:3657ba2e4d/model_best.pth")

domain_resnet = resnet18()
domain_fc = FDC(hidden_dim=512, cat_num=4, drop_xp=True).cuda()

new_dict = {}
for k in domain_model['state_dict_resnet'].keys():
    new_dict[k.replace("module.", "")] = domain_model['state_dict_resnet'][k]
    
domain_model['state_dict_resnet'] = new_dict
domain_resnet.load_state_dict(domain_model['state_dict_resnet'])

new_dict = {}
for k in domain_model['state_dict_classifier'].keys():
    new_dict[k.replace("module.", "")] = domain_model['state_dict_classifier'][k]
    
domain_model['state_dict_classifier'] = new_dict
domain_fc.load_state_dict(domain_model['state_dict_classifier'])
    
    


root_path = "/proj/vondrick3/james/pacs_data_cat"
train_dataset = MultiDomainLoader(dataset_root_dir=root_path,
                                      train_split=['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house'])
test_data = DomainTest(dataset_root_dir=root_path, test_split=['person'])
test_rand_data = RandomData(dataset_root_dir=root_path, all_split=['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house'])
test_d = "person"

val_data = test_data

print('domain', test_d)


train_sampler = None

eval_batch_size=32
workers = 4
samples = 10
val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=eval_batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=eval_batch_size, shuffle=False,
    num_workers=workers, pin_memory=True, sampler=train_sampler)
random_loader = torch.utils.data.DataLoader(
    test_rand_data, batch_size=eval_batch_size*samples, shuffle=True,
    num_workers=workers*2, pin_memory=True, sampler=train_sampler)

domain_resnet = domain_resnet.to('cuda')
domain_fc = domain_fc.to('cuda')

domain_resnet.eval()
domain_fc.eval()

test_robust_acc = 0
test_robust_acc_domain = 0
test_n = 0
domain_outs = []
gt_outs = []

with torch.no_grad():
    for i, bs_pair in enumerate(zip(test_loader, random_loader)):
        batch, batch_rand = bs_pair        
        X, y = batch
        Xp, _ = batch_rand
        X = X.cuda()
        y = y.cuda()
        Xp = Xp.cuda()
        _, feature = domain_resnet(X)
        bs_m = feature.size(0)
        j = 0
        logit_compose = domain_fc(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])
        for jj in range(samples-1):
            logit_compose = logit_compose + domain_fc(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])
        test_robust_acc_domain += (logit_compose.max(1)[1] == y).sum().item()
        test_n += y.size(0)
        domain_outs.extend(logit_compose.max(1)[1].detach().cpu().numpy().tolist())
        gt_outs.extend(y.detach().cpu().numpy().tolist())
        
print(domain_outs)
print(gt_outs)
print()
print(test_robust_acc_domain, test_n)
import pdb
pdb.set_trace()

    


