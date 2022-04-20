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
OOD_CAT = 0

d_models = {
    'art_painting': "/proj/james/AudioDefense_/Control/results/pacs_art_bl_A/2022-04-19_14:38:380c38766c/model_best.pth",
    'cartoon': "/proj/james/AudioDefense_/Control/results/pacs_cartoon_bl_C/2022-04-19_14:38:3971f0585f/model_best.pth",
    'sketch': "/proj/james/AudioDefense_/Control/results/pacs_sketch_bl_S/2022-04-19_13:14:5944e5b422/model_best.pth",
    'photo': "/proj/james/AudioDefense_/Control/results/pacs_photo_bl_P/2022-04-19_13:15:02d0d03c54/model_best.pth"
}
category_model = torch.load(d_models[test_d])

category_resnet_A = resnet18()
category_resnet_A.fc = nn.Linear(512, 7)

new_dict = {}
for k in category_model['state_dict_resnet'].keys():
    new_dict[k.replace("module.", "")] = category_model['state_dict_resnet'][k]
    

category_model['state_dict_resnet'] = new_dict
category_resnet_A.load_state_dict(category_model['state_dict_resnet'])



domain_model = torch.load("/proj/james/AudioDefense_/Control/results/CAT_FINAL_/2022-04-19_15:40:28355be6a1/model_best.pth")

domain_resnet = resnet18()
domain_resnet.fc = nn.Linear(512, 4)

new_dict = {}
for k in domain_model['state_dict_resnet'].keys():
    new_dict[k.replace("module.", "")] = domain_model['state_dict_resnet'][k]
    
# import pdb
# pdb.set_trace()
domain_model['state_dict_resnet'] = new_dict
domain_resnet.load_state_dict(domain_model['state_dict_resnet'])



pacs_categories = ['art_painting',  'cartoon', 'photo', 'sketch']
root_path = "/proj/james/pacs_data"
test_data = DomainTest(dataset_root_dir=root_path, test_split=[test_d])
test_rand_data = RandomData(dataset_root_dir=root_path, all_split=[x for x in pacs_categories if x!=test_d])

val_data = test_data
# print(val_data.class_to_idx)
# import pdb

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
domain_resnet = domain_resnet.to('cuda')

category_resnet_A.eval()
domain_resnet.eval()



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
        logit_compose = feature
        valid_images = (y == OOD_CAT)
        test_robust_acc_cat += (torch.logical_and(valid_images, logit_compose.max(1)[1] == y)).sum().item()
        test_n += valid_images.sum().item()
        
        batch, batch_rand = bs_pair
        X, y = batch
        Xp, _ = batch_rand
        X = X.cuda()
        y = y.cuda()
        Xp = Xp.cuda()
        _, feature = domain_resnet(X)
        bs_m = feature.size(0)
        j = 0
        logit_compose_domain = feature

        cat_y = torch.ones(y.size(0)) * test_val
        cat_y = cat_y.to('cuda')
        
        test_robust_acc_domain += (torch.logical_and(valid_images, logit_compose_domain.max(1)[1] == cat_y)).sum().item()
        domain_outs.extend(logit_compose_domain.max(1)[1].detach().cpu().numpy().tolist())
        
        test_robust_final += torch.logical_and(valid_images, torch.logical_and(logit_compose_domain.max(1)[1] == cat_y, logit_compose.max(1)[1] == y)).sum()
        
test_robust_final = test_robust_final.item()
# import pdb
# pdb.set_trace()
print(test_n)
print("Test Domain: ", test_d)
print("Category Acc. ", test_robust_acc_cat / test_n)
print("Domain Acc. ", test_robust_acc_domain / test_n)
print("Multi-task Acc. ", test_robust_final / test_n)
    


