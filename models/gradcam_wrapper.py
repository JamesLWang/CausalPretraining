import torch
import torch.nn as nn
import torch.nn.functional as F


from models.resnet import FDC5, resnet50, FConly
from models.resnet_wider_simclr import resnet50x4


class GradWrap(nn.Module):
    '''SimCLR Front door wrapper'''
    def __init__(self, latent_dim=8192, outdim=1000):
        super(GradWrap, self).__init__()

        self.backbone = resnet50x4()
        self.FDC = FDC5(hidden_dim=latent_dim, cat_num=outdim,drop_xp=False)

    def forward(self, x_concat):
        x = x_concat[0]
        x = x.unsqueeze(0)
        xp = x_concat[1:]
        _, fea = self.backbone(x)
        fea = fea

        pred = self.FDC(fea, xp, test=True, random_detach=True, noise_inside=True)
        return pred

class BackboneOnly(nn.Module):
    '''SimCLR baseline'''
    def __init__(self, latent_dim=8192, outdim=1000):
        super(BackboneOnly, self).__init__()

        self.backbone = resnet50x4()

    def forward(self, x):
        out, fea = self.backbone(x)
        return out


class GradSPWrap(nn.Module):
    '''Spurious, not in use'''
    def __init__(self, latent_dim=8192, outdim=1000):
        super(GradSPWrap, self).__init__()

        self.backbone = resnet50x4()
        self.FDC = FDC5(hidden_dim=latent_dim, cat_num=outdim,drop_xp=False)

    def forward(self, x):
        _, fea = self.backbone(x)
        fea = fea

        pred = self.FDC(fea, x, test=False, random_detach=True, noise_inside=True)
        return pred


class WaterBackboneOnly(nn.Module):
    '''Ours waterbird baseline'''
    def __init__(self, latent_dim=2048, outdim=2):
        super(WaterBackboneOnly, self).__init__()

        self.backbone = resnet50()
        self.fc = FConly(hidden_dim=latent_dim, out_dim=outdim)

    def forward(self, x):
        _, fea = self.backbone(x)
        prediction = self.fc(fea)
        return prediction

class WaterFDGradWrap(nn.Module):
    '''Waterbird Front door wrapper'''
    def __init__(self, latent_dim=2048, outdim=2):
        super(WaterFDGradWrap, self).__init__()

        self.backbone = resnet50()
        self.FDC = FDC5(hidden_dim=latent_dim, cat_num=outdim,drop_xp=False)

    def forward(self, x_concat):
        x = x_concat[0]
        x = x.unsqueeze(0)
        xp = x_concat[1:]
        _, fea = self.backbone(x)
        fea = fea

        pred = self.FDC(fea, xp, test=True, random_detach=True, noise_inside=True)
        return pred


class FDC5Only(nn.Module):
    '''Spurious branch'''
    def __init__(self, hidden_dim=2048, cat_num=1000, drop_xp=False, drop_xp_ratio=0.5, middle_hidden=1024):
        super(FDC5Only, self).__init__()
        sub_in_dim=64
        print("FDC 5")
        self.drop_xp = drop_xp

        self.conv1 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2, padding=0,
                               bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0,
                               bias=False)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0,
        #                        bias=False)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        # self.bn5 = nn.BatchNorm2d(512)
        # self.fc_z2 = nn.Linear(sub_in_dim, 64)
        self.dropout = nn.Dropout(p=drop_xp_ratio)
        # self.dropout2 = nn.Dropout(p=0.5)

        self.fc_c1 = nn.Linear(128, middle_hidden)
        # self.bn = nn.BatchNorm1d(1024)
        self.fc_c2 = nn.Linear(middle_hidden, cat_num)

    def forward(self, z2, detach=False, drop_xp=False):
        z2 = F.relu(self.bn1(self.conv1(z2)))
        z2 = F.relu(self.bn2(self.conv2(z2)))
        # z2 = self.dropout(z2)
        z2 = F.relu(self.bn3(self.conv3(z2)))
        # z2 = F.relu(self.bn4(self.conv4(z2)))
        # z2 = F.relu(self.bn5(self.conv5(z2)))
        z2 = F.avg_pool2d(z2, 14)
        z2 = z2.reshape(z2.size(0), -1)
        if detach:
            z2 = z2.detach()
        if self.drop_xp:
            z2 = self.dropout(z2)

        hh = F.relu(self.fc_c1(z2))
        out = self.fc_c2(hh)
        return out
