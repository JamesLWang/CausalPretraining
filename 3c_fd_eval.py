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

test_d = 'photo'

d_models = {
    'art_painting': "/proj/james/AudioDefense_/Control/results/pacs_art_A/2022-04-16_00:45:06ea32f3f9/model_best.pth",
    'cartoon': "/proj/james/AudioDefense_/Control/results/pacs_cartoon_C/2022-04-16_00:45:3043729215/model_best.pth",
    'sketch': "/proj/james/AudioDefense_/Control/results/pacs_sketch_S/2022-04-16_00:45:33b487de75/model_best.pth",
    'photo': "/proj/james/AudioDefense_/Control/results/pacs_photo_P/2022-04-16_01:01:4216bfd383/model_best.pth"
}
category_model = torch.load(d_models[test_d])

category_resnet_A = resnet18()
category_fc_A = FDC(hidden_dim=512, cat_num=7, drop_xp=True).cuda()

new_dict = {}
for k in category_model['state_dict_resnet'].keys():
    new_dict[k.replace("module.", "")] = category_model['state_dict_resnet'][k]
    
category_model['state_dict_resnet'] = new_dict
category_resnet_A.load_state_dict(category_model['state_dict_resnet'])

new_dict = {}
for k in category_model['state_dict_classifier'].keys():
    new_dict[k.replace("module.", "")] = category_model['state_dict_classifier'][k]
    
category_model['state_dict_classifier'] = new_dict
category_fc_A.load_state_dict(category_model['state_dict_classifier'])
    
    
domain_model = torch.load("/proj/james/AudioDefense_/Control/results/pacs_cat_/2022-04-16_01:05:3657ba2e4d/model_best.pth")

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
    

pacs_categories = ['art_painting',  'cartoon', 'photo', 'sketch']


root_path = "/proj/james/pacs_data"
test_data = DomainTest(dataset_root_dir=root_path, test_split=[test_d])
test_rand_data = RandomData(dataset_root_dir=root_path, all_split=[x for x in pacs_categories if x!=test_d])

val_data = test_data

test_val_dict = {
    'art_painting': 0,
    'cartoon': 1,
    'photo': 2,
    'sketch': 3
}

test_val = test_val_dict[test_d]

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

category_resnet_A = category_resnet_A.to('cuda')
category_fc_A = category_fc_A.to('cuda')

domain_resnet = domain_resnet.to('cuda')
domain_fc = domain_fc.to('cuda')

category_resnet_A.eval()
category_fc_A.eval()

domain_resnet.eval()
domain_fc.eval()



test_robust_acc_cat = 0
test_robust_acc_domain = 0
test_robust_final = 0
test_n = 0
domain_outs = []

with torch.no_grad():
    for i, bs_pair in enumerate(zip(test_loader, random_loader)):
        batch, batch_rand = bs_pair
        X, y = batch
        Xp, _ = batch_rand
        
        X = X.cuda()
        y = y.cuda()
        Xp = Xp.cuda()
        
        _, feature = category_resnet_A(X)
        bs_m = feature.size(0)
        j = 0
        logit_compose = category_fc_A(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])
        for jj in range(samples-1):
            logit_compose = logit_compose + category_fc_A(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])
            
        test_robust_acc_cat += (logit_compose.max(1)[1] == y).sum().item()
        test_n += y.size(0)
        
        batch, batch_rand = bs_pair
        X, y = batch
        Xp, _ = batch_rand
        X = X.cuda()
        y = y.cuda()
        Xp = Xp.cuda()
        _, feature = domain_resnet(X)
        bs_m = feature.size(0)
        j = 0
        logit_compose_domain = domain_fc(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])
        for jj in range(samples-1):
            logit_compose_domain = logit_compose_domain + domain_fc(feature, Xp[j*bs_m:(j+1)*bs_m, :, :, :])

        cat_y = torch.ones(y.size(0)) * test_val
        cat_y = cat_y.to('cuda')
        
        test_robust_acc_domain += (logit_compose_domain.max(1)[1] == cat_y).sum().item()
        domain_outs.extend(logit_compose_domain.max(1)[1].detach().cpu().numpy().tolist())
        
        test_robust_final += torch.logical_and(logit_compose_domain.max(1)[1] == cat_y, logit_compose.max(1)[1] == y).sum()
        
test_robust_final = test_robust_final.item()
# import pdb
# pdb.set_trace()
print("Test Domain: ", test_d)
print("Category Acc. ", test_robust_acc_cat / test_n)
print("Domain Acc. ", test_robust_acc_domain / test_n)
print("Multi-task Acc. ", test_robust_final / test_n)
    


