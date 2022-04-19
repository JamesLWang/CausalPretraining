'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet_backbone(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_backbone, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        mid_out = F.relu(self.bn(out))
        out = F.avg_pool2d(mid_out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, mid_out


class PreActResNet_Encoder(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_Encoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.bn = nn.BatchNorm2d(512 * block.expansion)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # mid_out = F.relu(self.bn(out))
        # out = F.avg_pool2d(mid_out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out



class FDC_deep_preact(nn.Module):
    def __init__(self, z_hidden, sub_in_dim, block=PreActBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(FDC_deep_preact, self).__init__()

        self.in_planes = 64

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2, bias=False) # hurt final result

        inner_dim = 128
        # self.fc_c1 = nn.Linear(z_hidden + inner_dim, z_hidden)
        self.fc_c2 = nn.Linear(z_hidden + inner_dim, 10)

        self.fc_z2 = nn.Linear(sub_in_dim, 512)
        self.fc2_z2 = nn.Linear(512, inner_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z1, z2, test=True):
        out = self.layer2(z1)
        out = self.layer3(out)
        out = self.layer4(out)
        z1 = F.relu(self.bn(out))
        # out = F.avg_pool2d(mid_out, 4)
        # z1 = out.view(out.size(0), -1)

        # print(z1.size(), 'z1')
        bs = z1.size(0)
        z1 = F.avg_pool2d(z1, 4)
        z1 = z1.reshape(bs, -1)

        # z2 = F.relu(self.conv1(z2))
        bs_fd = z2.size(0)
        z2 = z2.reshape(bs_fd, -1)
        z2 = F.relu(self.fc_z2(z2))
        z2 = F.relu(self.fc2_z2(z2))
        # TODO: maybe add fc to z2

        if test:
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs * bs, -1)

            # h3 = F.relu(self.fc_c1(hh))
            out = self.fc_c2(hh)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)

        else:
            hh = torch.cat((z1, z2), dim=1)
            # h3 = F.relu(self.fc_c1(h))
            out = self.fc_c2(hh)
            return out


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet18_backbone(num_classes=10):
    return PreActResNet_backbone(PreActBlock, [2,2,2,2], num_classes=num_classes)


def PreActResNet18_encoder(num_classes=10):
    return PreActResNet_Encoder(PreActBlock, [2,2,2,2], num_classes=num_classes)

def PreActResNet34():
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()


#
in_dim=512
z_hidden=64
class VAE_preact(nn.Module):
    def __init__(self):
        super(VAE_preact, self).__init__()

        self.fc21 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc22 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fc3 = nn.Linear(z_hidden, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, in_dim)

        # Decoder
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(512, 256, 3, 1)
        self.d22 = nn.Conv2d(256, 128, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(256, 1.e-3)
        self.bn62 = nn.BatchNorm2d(128, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(128, 64, 3, 1)
        self.d32 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(64, 1.e-3)
        self.bn72 = nn.BatchNorm2d(64, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(64, 32, 3, 1)
        self.d42 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32, 1.e-3)
        self.bn82 = nn.BatchNorm2d(32, 1.e-3)

        self.fc_last = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # self.fc_c1 = nn.Linear(z_hidden, z_hidden)
        # self.fc_c2 = nn.Linear(z_hidden, 2)

        self.relu = nn.ReLU()

        # self.embed = nn.Embedding(10, emb_dim)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # h1 = F.relu(self.fc2(h1))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

        h2 = F.relu(self.bn6(self.d2(self.pd1(self.up1(z)))))
        h2 = F.relu(self.bn62(self.d22(h2)))
        h3 = F.relu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h3 = F.relu(self.bn72(self.d32(h3)))
        h4 = F.relu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h4 = F.relu(self.bn82(self.d42(h4)))
        # h5 = F.relu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return torch.sigmoid(self.fc_last(h4))



    # def classifier(self, z):
    #     h3 = F.relu(self.fc_c1(z))
    #     return self.fc_c2(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar


class VAE_Small(nn.Module):
    def __init__(self):
        super(VAE_Small, self).__init__()

        self.fc21 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc22 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)
        # self.fc3 = nn.Linear(z_hidden, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, in_dim)

        # Decoder
        # self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd1 = nn.ReplicationPad2d(1)
        # self.d2 = nn.Conv2d(512, 256, 3, 1)
        # self.d22 = nn.Conv2d(256, 128, 3, 1, padding=1)
        # self.bn6 = nn.BatchNorm2d(256, 1.e-3)
        # self.bn62 = nn.BatchNorm2d(128, 1.e-3)
        #
        # self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd2 = nn.ReplicationPad2d(1)
        # self.d3 = nn.Conv2d(128, 64, 3, 1)
        # self.d32 = nn.Conv2d(64, 64, 3, 1, padding=1)
        # self.bn7 = nn.BatchNorm2d(64, 1.e-3)
        # self.bn72 = nn.BatchNorm2d(64, 1.e-3)
        #
        # self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        # self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(64, 32, 3, 1, padding=1)
        self.d42 = nn.Conv2d(32, 32, 3, 1, padding=1)
        self.bn8 = nn.BatchNorm2d(32, 1.e-3)
        self.bn82 = nn.BatchNorm2d(32, 1.e-3)

        self.fc_last = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # self.fc_c1 = nn.Linear(z_hidden, z_hidden)
        # self.fc_c2 = nn.Linear(z_hidden, 2)

        self.relu = nn.ReLU()

        # self.embed = nn.Embedding(10, emb_dim)

    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        # h1 = F.relu(self.fc2(h1))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

        # h2 = F.relu(self.bn6(self.d2(self.pd1(self.up1(z)))))
        # h2 = F.relu(self.bn62(self.d22(h2)))
        # h3 = F.relu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        # h3 = F.relu(self.bn72(self.d32(h3)))
        h4 = F.relu(self.bn8(self.d4(z)))
        h4 = F.relu(self.bn82(self.d42(h4)))
        # h5 = F.relu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return torch.sigmoid(self.fc_last(h4))



    # def classifier(self, z):
    #     h3 = F.relu(self.fc_c1(z))
    #     return self.fc_c2(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        return self.decode(z), mu, logvar







class FDC_preact(nn.Module):
    def __init__(self, z_hidden, sub_in_dim):
        super(FDC_preact, self).__init__()

        inner_dim = 32
        self.fc_c1 = nn.Linear(z_hidden + inner_dim, z_hidden)
        self.fc_c2 = nn.Linear(z_hidden, 10)

        self.fc_z2 = nn.Linear(sub_in_dim, 32)

    def forward(self, z1, z2, test=True):
        bs = z1.size(0)
        z1 = F.avg_pool2d(z1, 4)
        z1 = z1.reshape(bs, -1)

        bs_fd = z2.size(0)
        z2 = z2.reshape(bs_fd, -1)
        z2 = F.relu(self.fc_z2(z2))
        # TODO: maybe add fc to z2

        if test:
            z1 = z1.unsqueeze(1)
            z1 = z1.repeat(1, bs, 1)
            z2 = z2.unsqueeze(0)
            z2 = z2.repeat(bs, 1, 1)

            hh = torch.cat((z1, z2), dim=2)

            hh = hh.view(bs * bs, -1)

            h3 = F.relu(self.fc_c1(hh))
            out = self.fc_c2(h3)
            out = out.view(bs, bs, -1)

            return torch.sum(out, dim=1)

        else:
            h = torch.cat((z1, z2), dim=1)
            h3 = F.relu(self.fc_c1(h))
            out = self.fc_c2(h3)
            return out

