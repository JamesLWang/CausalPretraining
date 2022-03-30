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
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        # normalize
    ]),
}


image_datasets = {
    'train_object': 
    datasets.ImageFolder('/home/jlw2247/vondrick_2/OfficeHome_Train_Object', data_transforms['train']),
    'test_object': 
    datasets.ImageFolder('/home/jlw2247/vondrick_2/OfficeHome_Test_Object', data_transforms['validation']),
    'train_domain': 
    datasets.ImageFolder('/home/jlw2247/vondrick_2/OfficeHome_Train_Domain', data_transforms['train']),
    'test_domain': 
    datasets.ImageFolder('/home/jlw2247/vondrick_2/OfficeHome_Test_Domain', data_transforms['validation'])
    
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

class AlexNet_OH_DOM(nn.Module):
    def __init__(self):
        super(AlexNet_OH_DOM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 6400, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 128)
        self.fc3 = nn.Linear(in_features=128 , out_features=4)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda

model = AlexNet_OH_DOM() #to compile the model
model = model.to(device=device) #to send the model for training on either cuda or cpu

## Loss and optimizer
learning_rate = 1e-4 #I picked this because it seems to be the most used by experts
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate) #Adam seems to be the most popular for deep learning

EPOCHS = 200
for epoch in range(EPOCHS):
    loss_ep = 0
    
    for batch_idx, (data, targets) in enumerate(dataloaders['train_domain']):
        data = data.to(device=device)
        targets = targets.to(device=device)

        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores,targets)
        
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
    print(f"Loss in epoch {epoch} :::: {loss_ep/batch_size}")
    
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data,targets) in enumerate(dataloaders['test_domain']):
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

torch.save(model.state_dict(), "1a_Resnet18_dom.pth")
