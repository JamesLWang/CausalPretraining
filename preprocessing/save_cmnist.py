import os
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from torch.nn import functional as F

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from torchvision.utils import save_image

mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

save_root="/local/vondrick/cz/CMNIST"
os.makedirs(save_root, exist_ok=True)


def make_environment(imgs, labels):
    labels = labels.float()
    imgs = imgs.reshape((-1, 28, 28))# [:, ::2, ::2]

    imgs2 = torch.stack([imgs, imgs, imgs], dim=1)
    imgs = torch.stack([imgs, imgs, imgs], dim=1)


    imgs[labels == 0, 0, :, :] = torch.ones_like(imgs[labels == 0, 0, :, :])

    imgs[labels == 0, 1, :, :] = torch.ones_like(imgs[labels == 0, 1, :, :])

    imgs[labels == 1, 1, :, :] = torch.ones_like(imgs[labels == 1, 1, :, :])
    imgs[labels == 1, 2, :, :] = torch.ones_like(imgs[labels == 1, 2, :, :])

    imgs[labels == 2, 2, :, :] = torch.ones_like(imgs[labels == 2, 2, :, :])
    imgs[labels == 2, 0, :, :] = torch.ones_like(imgs[labels == 2, 0, :, :])

    imgs[labels == 3, 0, :, :] = torch.ones_like(imgs[labels == 3, 0, :, :])

    imgs[labels == 4, 1, :, :] = torch.ones_like(imgs[labels == 4, 1, :, :])

    imgs[labels == 5, 2, :, :] = torch.ones_like(imgs[labels == 5, 2, :, :])

    imgs[labels == 6, 0, :, :] = torch.zeros_like(imgs[labels == 6, 0, :, :])
    imgs[labels == 6, 1, :, :] = torch.zeros_like(imgs[labels == 6, 1, :, :])

    imgs[labels == 7, 1, :, :] = torch.zeros_like(imgs[labels == 7, 1, :, :])
    imgs[labels == 7, 2, :, :] = torch.zeros_like(imgs[labels == 7, 2, :, :])

    imgs[labels == 8, 2, :, :] = torch.zeros_like(imgs[labels == 8, 2, :, :])
    imgs[labels == 8, 0, :, :] = torch.zeros_like(imgs[labels == 8, 0, :, :])

    imgs[labels == 9, 1, :, :] = torch.zeros_like(imgs[labels == 9, 1, :, :])



    ######
    imgs2[labels == 5, 0, :, :] = torch.ones_like(imgs2[labels == 5, 0, :, :])
    imgs2[labels == 5, 1, :, :] = torch.zeros_like(imgs2[labels == 5, 1, :, :])

    imgs2[labels == 3, 1, :, :] = torch.ones_like(imgs2[labels == 3, 1, :, :])
    imgs2[labels == 3, 2, :, :] = torch.zeros_like(imgs2[labels == 3, 2, :, :])

    imgs2[labels == 4, 2, :, :] = torch.ones_like(imgs2[labels == 4, 2, :, :])
    imgs2[labels == 4, 0, :, :] = torch.zeros_like(imgs2[labels == 4, 0, :, :])

    imgs2[labels == 1, 0, :, :] = torch.zeros_like(imgs2[labels == 1, 0, :, :])

    imgs2[labels == 2, 1, :, :] = torch.zeros_like(imgs2[labels == 2, 1, :, :])

    imgs2[labels == 0, 2, :, :] = torch.zeros_like(imgs2[labels == 0, 2, :, :])

    imgs2[labels == 6, 0, :, :] = torch.zeros_like(imgs2[labels == 6, 0, :, :])
    imgs2[labels == 6, 1, :, :] = torch.ones_like(imgs2[labels == 6, 1, :, :])

    imgs2[labels == 7, 1, :, :] = torch.zeros_like(imgs2[labels == 7, 1, :, :])
    imgs2[labels == 7, 2, :, :] = torch.ones_like(imgs2[labels == 7, 2, :, :])

    imgs2[labels == 8, 2, :, :] = torch.zeros_like(imgs2[labels == 8, 2, :, :])
    imgs2[labels == 8, 0, :, :] = torch.ones_like(imgs2[labels == 8, 0, :, :])

    imgs2[labels == 9, 1, :, :] = torch.ones_like(imgs2[labels == 9, 1, :, :])
    #######

    print(imgs.size(), imgs2.size())
    imgs = torch.cat([imgs, imgs2], dim=0)

    print(labels.size())
    labels = torch.cat([labels, labels], dim=0)

    img0 = imgs[labels == 0]
    img1 = imgs[labels == 1]
    img2 = imgs[labels == 2]
    img3 = imgs[labels == 3]
    img4 = imgs[labels == 4]
    img5 = imgs[labels == 5]
    img6 = imgs[labels == 6]
    img7 = imgs[labels == 7]
    img8 = imgs[labels == 8]
    img9 = imgs[labels == 9]

    pair = {0: img0, 1: img1, 2: img2, 3: img3, 4: img4, 5: img5, 6: img6, 7: img7, 8: img8, 9: img9}
    for k in pair.keys():
        pair[k] = (pair.get(k).float() / 255.).cuda()

    return (imgs.float() / 255.).cuda(), labels[:, None].cuda(), pair


def test_env(test_imgs, labels):
    labels = labels.float()
    test_imgs = test_imgs.reshape((-1, 28, 28))#[:, ::2, ::2]
    test_imgs = torch.stack([test_imgs, test_imgs, test_imgs], dim=1)

    total = test_imgs.size(0)
    ee=20
    for cnt in range(total//ee):
        channel = np.random.choice(3, 2)
        color = np.random.sample([3]) > 0.5
        for ch in channel:
            if color[ch]:
                test_imgs[cnt*ee:(cnt+1)*ee, ch, :, :] = torch.ones_like(test_imgs[cnt*ee:(cnt+1)*ee, ch, :, :])
            else:
                test_imgs[cnt*ee:(cnt+1)*ee, ch, :, :] = torch.zeros_like(test_imgs[cnt*ee:(cnt+1)*ee, ch, :, :])
    return (test_imgs.float() / 255.).cuda(), labels[:, None].cuda()



train_set, train_label, pair = make_environment(mnist_train[0], mnist_train[1])
# i2, l2 = make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1)

print(train_set.min(), 'min')

i3, l3 = test_env(mnist_val[0], mnist_val[1])

test_c, test_l_c, _ = make_environment(mnist_val[0], mnist_val[1])

source_path=os.path.join(save_root, "source")
os.makedirs(source_path, exist_ok=True)

f1=open(os.path.join(save_root, 'source_train.txt'), "w")

for i in range(10):
    os.makedirs(os.path.join(source_path, str(i)), exist_ok=True)

for ii in range(train_set.size(0)):
    save_image(train_set[ii, :, :, :], os.path.join(source_path, str(int(train_label[ii, 0].item())),
               'train_{}.png'.format(ii)), normalize=True)
    f1.write(os.path.join(source_path, str(int(train_label[ii, 0].item())),
               'train_{}.png'.format(ii)) + ' ' + str(int(train_label[ii, 0].item()))+'\n')
f1.close()


source_path=os.path.join(save_root, "source_val")
os.makedirs(source_path, exist_ok=True)
f1=open(os.path.join(save_root, 'source_test.txt'), "w")
for i in range(10):
    os.makedirs(os.path.join(source_path, str(i)), exist_ok=True)

for ii in range(test_c.size(0)):
    save_image(test_c[ii, :, :, :], os.path.join(source_path, str(int(test_l_c[ii, 0].item())),
               'train_{}.png'.format(ii)), normalize=True)
    f1.write(os.path.join(source_path, str(int(train_label[ii, 0].item())),
                          'train_{}.png'.format(ii)) + ' ' + str(int(train_label[ii, 0].item()))+'\n')
f1.close()


source_path=os.path.join(save_root, "target")
os.makedirs(source_path, exist_ok=True)
f1=open(os.path.join(save_root, 'target_test.txt'), "w")
for i in range(10):
    os.makedirs(os.path.join(source_path, str(i)), exist_ok=True)

for ii in range(i3.size(0)):
    save_image(i3[ii, :, :, :], os.path.join(source_path, str(int(l3[ii, 0].item())),
               'train_{}.png'.format(ii)), normalize=True)
    if not os.path.exists(os.path.join(source_path, str(int(l3[ii, 0].item())), 'train_{}.png'.format(ii))):
        print('not exists', os.path.join(source_path, str(int(l3[ii, 0].item())), 'train_{}.png'.format(ii)))
        continue
    f1.write(os.path.join(source_path, str(int(l3[ii, 0].item())), 'train_{}.png'.format(ii))
             + ' ' + str(int(train_label[ii, 0].item()))+'\n')
f1.close()
