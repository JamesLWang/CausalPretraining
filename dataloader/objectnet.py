import torch, os
import numpy as np
from PIL import Image

import torch
import json
from torchvision import transforms
import torchvision.transforms as transforms


class ObjectNetLoader(torch.utils.data.Dataset):

    def __init__ (self,
                  train_base_dir, few_test=None, composed_transform=None, center_crop=False
                  ):
        super().__init__()

        self.train_path = train_base_dir
        self.categories_list = os.listdir(self.train_path)
        self.categories_list.sort()

        self.file_lists = []
        self.label_lists = []
        self.few_test = few_test
        if composed_transform is None:
             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

             composed_transform = transforms.Compose([
                transforms.Compose([
                transforms.Resize(256+32) if not center_crop else transforms.Resize(int(256 * 1.4)),
                transforms.CenterCrop(256 if not center_crop else 224),
                transforms.ToTensor(),
                normalize,
                ])
                ])
        self.composed_transforms=composed_transform

        with open('preprocessing/obj2imgnet_id.txt') as f:
            self.dict_obj2imagenet_id = json.load(f)

        for each in self.categories_list:
            folder_path = os.path.join(self.train_path, each)

            files_names = os.listdir(folder_path)

            for eachfile in files_names:
                image_path = os.path.join(folder_path, eachfile)
                self.file_lists.append(image_path)
                # print("label_length={}".format(len(self.dict_obj2imagenet_id[each])))
                label_len = len(self.dict_obj2imagenet_id[each])
                self.label_lists.append(self.dict_obj2imagenet_id[each]+[-1]*(11-label_len))  # since the loader cutoff automatically on the
                # minimum length of labels, we make the minimum to be 11, by adding redundant 10 -1s.

    def __len__(self):
        if self.few_test is not None:
            return self.few_test
        else:
            return len(self.label_lists)

    def _transform(self, sample):
        return self.composed_transforms(sample)

    def __getitem__(self, item):
        path_list=self.file_lists[item]
        img = Image.open(path_list).convert("RGB")

        img_tensor = self._transform(img)
        img.close()
        labels = self.label_lists[item]
        return {"images": img_tensor, "labels": labels, "path": path_list}