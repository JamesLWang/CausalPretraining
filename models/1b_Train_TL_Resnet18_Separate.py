import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pathlib
import torch.nn.functional as F
from torchvision.models.resnet import resnet18 as resnet18

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # normalize
        ]),
    'validation':
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalize
        ]),
}

root = pathlib.Path(__file__).parent.parent.resolve()
dataset_root = f"{root}/datasets/OfficeHome"
model_save_dir = f"{root}/model_checkpoints"

image_datasets = {
    'train_object':
        datasets.ImageFolder(f'{dataset_root}/OfficeHome_Train_Object', data_transforms['train']),
    'test_object':
        datasets.ImageFolder(f'{dataset_root}/OfficeHome_Test_Object', data_transforms['validation']),
    'train_domain':
        datasets.ImageFolder(f'{dataset_root}/OfficeHome_Train_Domain', data_transforms['train']),
    'test_domain':
        datasets.ImageFolder(f'{dataset_root}/OfficeHome_Test_Domain', data_transforms['validation'])

}

batch_size = 1296
dataloaders = {
    'train_object':
        torch.utils.data.DataLoader(image_datasets['train_object'],
                                    batch_size=batch_size,
                                    shuffle=True, num_workers=4),
    'test_object':
        torch.utils.data.DataLoader(image_datasets['test_object'],
                                    batch_size=32,
                                    shuffle=False, num_workers=4),
    'train_domain':
        torch.utils.data.DataLoader(image_datasets['train_domain'],
                                    batch_size=batch_size,
                                    shuffle=True, num_workers=4),
    'test_domain':
        torch.utils.data.DataLoader(image_datasets['test_domain'],
                                    batch_size=batch_size,
                                    shuffle=False, num_workers=4),

}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # training with either cpu or cuda

model = resnet18(pretrained=False)
model = model.to(device=device) #to send the model for training on either cuda or cpu
model = nn.DataParallel(model).cuda()

## Loss and optimizer
learning_rate = 1e-4  # I picked this because it seems to be the most used by experts
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam seems to be the most popular for deep learning

EPOCHS = 200
for epoch in range(EPOCHS):
    loss_ep = 0
    train_num_correct = 0
    train_num_samples = 0
    for batch_idx, (data, targets) in enumerate(dataloaders['train_object']):
        data = data.to(device=device)
        targets = targets.to(device=device)

        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores, targets)

        loss.backward()
        optimizer.step()
        loss_ep += loss.item()

        # check training acc
        _, predictions = scores.max(1)
        train_num_correct += (predictions == targets).sum()
        train_num_samples += predictions.size(0)
    print(f"Loss in epoch {epoch} :::: {loss_ep / batch_size}, train acc ::: {train_num_correct/float(train_num_samples):.2f}")

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data, targets) in enumerate(dataloaders['test_object']):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )

torch.save(model.state_dict(), f"{model_save_dir}/1b_Resnet18_obj.pth")