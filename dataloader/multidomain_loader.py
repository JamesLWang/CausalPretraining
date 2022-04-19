import torch.utils.data as data

import os, shutil
import sys
import time

import random
import argparse
import numpy as np
import glob
import matplotlib.pyplot as plt

from PIL import Image

import torch
from torchvision import transforms
import torchvision.transforms as transforms

import torch
from torch import nn
from torchvision.transforms import transforms

np.random.seed(0)


class RandomLoader(data.Dataset):
    def __init__(self, path, composed_transforms=None):
        super().__init__()
        self.path = path
        self.categories = os.listdir(path)
        self.categories.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories)}

        self.transform = composed_transforms

    def __getitem__(self, item):

        select_category = random.sample(self.categories, 1)[0]
        cat_path = os.path.join(self.path, select_category)
        img_lists = os.listdir(cat_path)
        select_img = random.sample(img_lists, 1)[0]
        while True:
            if '.png' in select_img:
                break
            else:
                select_img = random.sample(img_lists, 1)[0]

        img_path = os.path.join(cat_path, select_img)

        target = self.category2id[select_category]

        # path, target = self.filelist1[item % self.len1]  # need a shuffle to guarantee randomness
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert("RGB")
        sample = self._transform(img)
        img.close()
        # sample_imgnet, target_imgnet,
        return sample, target

    def __len__(self):
        return 1000000000

    def _transform(self, sample):
        return self.transform(sample)

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class DomainTest(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, test_split, subsample=1, noNormalize=False, fd_specified_preprocess=None, add_val=False, add_info=''):
        self.subsample = subsample
        self.test_split = test_split
        self.dataset_root_dir = dataset_root_dir
        self.add_val = add_val
        self.add_info = add_info

        tr_example_path = os.path.join(dataset_root_dir, test_split[0])
        if add_val:
            tr_example_path = os.path.join(tr_example_path, 'val')
        if add_info!='':
            tr_example_path = os.path.join(tr_example_path, add_info)

        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data = self.make_dataset()

        if noNormalize:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if fd_specified_preprocess is not None:
            self.transform = fd_specified_preprocess

    def make_dataset(self):
        all_data=[]
        for each in self.test_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            if self.add_val:
                domain_path = os.path.join(domain_path, 'val')
            if self.add_info != '':
                domain_path = os.path.join(domain_path, self.add_info)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    all_data.append([os.path.join(cate_path, img), cate])
        return all_data[::self.subsample]

    def __getitem__(self, index):
        img_path, cate = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, self.category2id[cate]

    def __len__(self):
        return len(self.all_data)

        # TODO: the data comes in also needs to be paired during testing!


class DomainTestPath(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, test_split, subsample=1, noNormalize=False, fd_specified_preprocess=None, add_val=False, add_info=''):
        self.subsample = subsample
        self.test_split = test_split
        self.dataset_root_dir = dataset_root_dir
        self.add_val = add_val
        self.add_info = add_info

        tr_example_path = os.path.join(dataset_root_dir, test_split[0])
        if add_val:
            tr_example_path = os.path.join(tr_example_path, 'val')
        if add_info!='':
            tr_example_path = os.path.join(tr_example_path, add_info)

        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data = self.make_dataset()

        if noNormalize:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        if fd_specified_preprocess is not None:
            self.transform = fd_specified_preprocess

    def make_dataset(self):
        all_data=[]
        for each in self.test_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            if self.add_val:
                domain_path = os.path.join(domain_path, 'val')
            if self.add_info != '':
                domain_path = os.path.join(domain_path, self.add_info)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    all_data.append([os.path.join(cate_path, img), cate])
        return all_data[::self.subsample]

    def __getitem__(self, index):
        img_path, cate = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, self.category2id[cate], img_path

    def __len__(self):
        return len(self.all_data)

        # TODO: the data comes in also needs to be paired during testing!

class DomainPCALoader(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, test_split):
        self.test_split = test_split
        self.dataset_root_dir = dataset_root_dir

        tr_example_path = os.path.join(dataset_root_dir, test_split[0])
        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data = self.make_dataset()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def make_dataset(self):
        all_data=[]
        for each in self.test_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    all_data.append([os.path.join(cate_path, img), cate])
        return all_data

    def __getitem__(self, index):
        img_path, cate = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, self.category2id[cate], img_path

    def __len__(self):
        return len(self.all_data)




class RandomData(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, all_split, noNormalize=False, fd_specified_preprocess=None, add_val=False):
        self.all_split = all_split
        self.dataset_root_dir = dataset_root_dir
        self.add_val = add_val

        tr_example_path = os.path.join(dataset_root_dir, all_split[0])
        if add_val:
            tr_example_path = os.path.join(tr_example_path, 'val')
        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()
        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}
        self.all_data = self.make_dataset()
        self.index_list = [i for i in range(len(self.all_data))]

        if noNormalize:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if fd_specified_preprocess is not None:
            self.transform = fd_specified_preprocess
        # self.transform = transforms.Compose([
        #     # transforms.Resize((224,224)),
        #     transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        #     transforms.RandomGrayscale(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        #TODO: try transform aug

    def make_dataset(self):
        all_data=[]
        for each in self.all_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            if self.add_val:
                domain_path = os.path.join(domain_path, 'val')
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    all_data.append([os.path.join(cate_path, img), cate])
        return all_data

    def __getitem__(self, index):
        index_id = random.choice(self.index_list)
        img_path, cate = self.all_data[index_id]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.transform(img_x)
        return img_x, self.category2id[cate]

    def __len__(self):
        return len(self.all_data)*20


class MultiDomainLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, train_split, subsample=1, bd_aug=False, fd_aug=True, noNormalize=False, fd_specified_preprocess=None):
        self.subsample = subsample
        self.train_split = train_split
        self.dataset_root_dir = dataset_root_dir
        if fd_aug:
            if noNormalize:
                self.augment_transform = transforms.Compose([
                    # transforms.Resize((224,224)),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                ])
            else:
                self.augment_transform = transforms.Compose([
                    # transforms.Resize((224,224)),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            if noNormalize:
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ])
            else:
                self.augment_transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        if bd_aug:
            if noNormalize:
                self.transform = transforms.Compose([
                    # transforms.Resize((224,224)),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                ])
            else:
                self.transform=transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        else:
            if noNormalize:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        if fd_specified_preprocess is not None:
            self.augment_transform = fd_specified_preprocess
            self.transform = fd_specified_preprocess

        tr_example_path = os.path.join(dataset_root_dir, train_split[0])
        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data, self.all_cate = self.make_dataset()


    def make_dataset(self):
        all_data=[]
        cnt=0
        for each in self.train_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    cnt+=1
                    if cnt==self.subsample:
                        all_data.append([os.path.join(cate_path, img), cate, each])
                        cnt=0

        all_cate=[[] for _ in self.categories_list]
        for d in all_data:
            each, tmp_cat, _ = d
            id = self.category2id[tmp_cat]
            all_cate[id].append(each)

        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, cate, domain = self.all_data[index]
        img_x = Image.open(img_path).convert("RGB")
        img_x = self.augment_transform(img_x)

        label = self.category2id[cate]

        # rd = random.sample(self.train_split, 1)[0]
        # selected_path = os.path.join(self.dataset_root_dir, rd, cate)
        #
        # img_xp_path = os.path.join(selected_path, random.sample(os.listdir(selected_path), 1)[0])

        id = self.category2id[cate]
        img_xp_path = random.sample(self.all_cate[id], 1)[0]

        img_xp = Image.open(img_xp_path).convert("RGB")
        img_xp = self.transform(img_xp)

        return img_x, img_xp, label






class MultiDomainLoaderTriple(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, train_split, subsample=1, bd_aug=False, noNormalize=False, ):
        self.subsample = subsample
        self.train_split = train_split
        self.dataset_root_dir = dataset_root_dir
        if noNormalize:
            self.augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor()
            ])
        else:
            self.augment_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        #SimCLR augmentation # this is bad
        s=1
        size=224
        # color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        # self.augment_transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                       transforms.RandomGrayscale(p=0.2),
        #                                       GaussianBlur(kernel_size=int(0.1 * size)),
        #                                       transforms.ToTensor()])
        # print('use SimCLR augmentation')

        tr_example_path = os.path.join(dataset_root_dir, train_split[0])
        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data, self.all_cate = self.make_dataset()


    def make_dataset(self):
        all_data=[]
        cnt=0
        for each in self.train_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    cnt+=1
                    if cnt==self.subsample:
                        all_data.append([os.path.join(cate_path, img), cate, each])
                        cnt=0

        all_cate=[[] for _ in self.categories_list]
        for d in all_data:
            each, tmp_cat, _ = d
            id = self.category2id[tmp_cat]
            all_cate[id].append(each)

        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, cate, domain = self.all_data[index]
        img_x_ori = Image.open(img_path).convert("RGB")
        img_x = self.augment_transform(img_x_ori)
        img_x_2 = self.augment_transform(img_x_ori)
        img_x_3 = self.augment_transform(img_x_ori)
        img_x_4 = self.augment_transform(img_x_ori)

        return img_x, img_x_2, img_x_3, img_x_4




class MultiDomainLoaderTripleFD(torch.utils.data.Dataset):
    def __init__(self, dataset_root_dir, train_split, subsample=1, bd_aug=False):
        self.subsample = subsample
        self.train_split = train_split
        self.dataset_root_dir = dataset_root_dir
        self.augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        tr_example_path = os.path.join(dataset_root_dir, train_split[0])
        self.categories_list = os.listdir(tr_example_path)

        self.categories_list.sort()

        self.category2id = {filename: fileintkey for fileintkey, filename in enumerate(self.categories_list)}

        self.all_data, self.all_cate = self.make_dataset()


    def make_dataset(self):
        all_data=[]
        cnt=0
        for each in self.train_split:
            domain_path = os.path.join(self.dataset_root_dir, each)
            for cate in self.categories_list:
                cate_path = os.path.join(domain_path, cate)
                for img in os.listdir(cate_path):
                    cnt+=1
                    if cnt==self.subsample:
                        all_data.append([os.path.join(cate_path, img), cate, each])
                        cnt=0

        all_cate=[[] for _ in self.categories_list]
        for d in all_data:
            each, tmp_cat, _ = d
            id = self.category2id[tmp_cat]
            all_cate[id].append(each)

        return all_data, all_cate

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        img_path, cate, domain = self.all_data[index]
        img_x_ori = Image.open(img_path).convert("RGB")
        img_x = self.augment_transform(img_x_ori)
        img_x_2 = self.augment_transform(img_x_ori)
        img_x_3 = self.augment_transform(img_x_ori)
        img_x_4 = self.augment_transform(img_x_ori)

        label = self.category2id[cate]

        id = self.category2id[cate]
        img_xp_path = random.sample(self.all_cate[id], 1)[0]

        img_xp = Image.open(img_xp_path).convert("RGB")
        img_xp = self.transform(img_xp)

        return img_x, img_x_2, img_x_3, img_x_4, img_xp, label
