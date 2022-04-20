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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='PreActResNet18')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--eval-batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
    parser.add_argument('--lr-max', default=5e-5, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--samples', default=10, type=int)
    parser.add_argument('--pgd-alpha', default=2, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='pacs', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--linear', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--drop_xp', action='store_true')
    parser.add_argument('--detach_xp', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    parser.add_argument('--style', default='', type=str, choices=['A', 'C', 'P', 'S'])
    parser.add_argument('--chkpt-iters', default=10, type=int)
    if 'cv' in socket.gethostname():
        parser.add_argument('--save_root_path', default='/proj/james/AudioDefense_/Control/results', type=str)
    else:
        parser.add_argument('--save_root_path', default='/proj/james/AudioDefense_/Control/results', type=str)
    return parser.parse_args()

def main():
    adda_times=1

    args = get_args()
    import uuid
    import datetime
    unique_str = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')

    args.fname = os.path.join(args.save_root_path, args.fname+'_'+args.style, timestamp + unique_str)
    if not os.path.exists(args.fname):
        os.makedirs(args.fname)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(args.fname, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    from dataloader.multidomain_loader import MultiDomainLoader, DomainTest, RandomData
    
    root_path = "/proj/james/pacs_data_cat"

    # subset from target domain validation, remember to also change the loader for validation loop

    train_dataset = MultiDomainLoader(dataset_root_dir=root_path,
                                      train_split=['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house'])
    test_data = DomainTest(dataset_root_dir=root_path,
                           test_split=['person'])
    test_rand_data = RandomData(dataset_root_dir=root_path,
                                all_split=['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house'])
    test_d = "person"

    val_data = test_data

    print('domain', test_d)

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    random_loader = torch.utils.data.DataLoader(
        test_rand_data, batch_size=args.eval_batch_size*args.samples, shuffle=True,
        num_workers=args.workers*2, pin_memory=True, sampler=train_sampler)

    from models.resnet import resnet18, FCClassifier, resnet50, FDC, FC2Classifier, FDC5

    resnet = resnet18()    
    resnet.fc = nn.Linear(512, 4)
    resnet = nn.DataParallel(resnet).cuda()
    resnet.train()



    params =  list(resnet.parameters())  # list(resnet.parameters()) +
    opt = torch.optim.Adam(params, lr=args.lr_max)

    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    best_test_robust_acc = 0
    best_val_robust_acc = 0

    start_epoch = 1

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info(
        'Epoch \t Train Time \t Test Time \t LR \t \t Train Loss \t Train Acc \t Train Robust Loss \t Train Robust Acc \t Test Loss \t Test Acc \t Test Robust Loss \t Test Robust Acc')
    epochs = 50



    for epoch in range(start_epoch, epochs+1):
        if args.train_all and epoch>1:
            resnet.train()
        else:
            resnet.eval()

        def freeze_bn(network):
            for m in network.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        freeze_bn(resnet)


        start_time = time.time()
        train_loss = 0
        train_acc = 0

        train_n = 0
        for i, batch in enumerate(train_loader):
            if args.eval:
                break
            X, Xp, y = batch
            X = X.cuda()
            Xp = X.cuda()
            y = y.cuda()
            
            _, prediction = resnet(X)
            cl_loss = criterion(prediction, y)
            loss = cl_loss
            train_loss += loss.item() * y.size(0)
            train_acc += (prediction.max(1)[1] == y).sum().item()

            train_n += y.size(0)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print("L1", L1_loss.item(), "cl", cl_loss.item())

        # if args.fast and (epoch % 1>0):
        #     print("train acc", train_acc/train_n)
        #     continue

        print('start eval')
        train_time = time.time()

        # classifier.eval()
        resnet.eval()

        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        with torch.no_grad():
            for i, bs_pair in enumerate(zip(val_loader, random_loader)):
                batch, batch_rand = bs_pair
                X, y = batch
                Xp, _ = batch_rand

                X = X.cuda()
                y = y.cuda()
                Xp = Xp.cuda()


                _, prediction = resnet(X)
                
                logit_compose = prediction

                test_robust_acc += (logit_compose.max(1)[1] == y).sum().item()
                test_robust_loss += loss.item() * y.size(0)
                test_n += y.size(0)

                if i>1 and args.fast:
                    break

        train_time = time.time()

        resnet.eval()

        # test_loss = 0
        # test_acc = 0
        # test_robust_loss = 0
        test_vanilla_acc = 0
        test_vanilla_loss=0
        test_n_v = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                X, y = batch

                X = X.cuda()
                y = y.cuda()

                _, prediction = resnet(X)
        
                bs_m = prediction.size(0)
                j = 0
                logit_compose = prediction

                # for jj in range(args.samples - 1):
                #     p = 0.5
                #     binomial = torch.distributions.binomial.Binomial(probs=1 - p)
                #     feature = feature * binomial.sample(feature.size()).cuda() #* (1.0 / (1 - p))
                #     logit_compose = logit_compose + classifier(feature, X, test=False)  # TODO:

                test_vanilla_acc += (logit_compose.max(1)[1] == y).sum().item()
                test_vanilla_loss += loss.item() * y.size(0)
                test_n_v += y.size(0)

                if i>1 and args.fast:
                    break


        test_time = time.time()

        print("epoch", epoch, "test domain", test_d,  " train acc", train_acc/train_n,
              "test vanilla", test_vanilla_acc/test_n_v, "test fd", test_robust_acc/test_n)



        # print("vanilla original test", test_vanilla_acc/test_n, "vanilla confounded test", test_confounded_vanilla_acc/test_n)

        if (epoch + 1) % args.chkpt_iters == 0 or epoch + 1 == epochs:
            torch.save({'classifier': resnet.state_dict()},
                       os.path.join(args.fname, f'model_{epoch}.pth'))
            torch.save(opt.state_dict(), os.path.join(args.fname, f'opt_{epoch}.pth'))

            # save best
        if test_robust_acc / test_n > best_test_robust_acc:
            torch.save({
                # 'state_dict_classifier': classifier.state_dict(),
                'state_dict_resnet': resnet.state_dict(),
                'test_robust_acc': test_robust_acc / test_n,
                'test_robust_loss': test_robust_loss / test_n,
                'test_loss': test_vanilla_loss / test_n,
                'test_acc': test_vanilla_acc / test_n,
            }, os.path.join(args.fname, f'model_best.pth'))
            best_test_robust_acc = test_robust_acc / test_n
            pair_vanilla_acc = test_vanilla_acc/test_n_v,

    print("best robust acc", best_test_robust_acc, "vanilla acc", pair_vanilla_acc, "\n\n\n")

    checkpoint = torch.load(os.path.join(args.fname, f'model_best.pth'))

    # checkpoint = torch.load('./resnet50.pth')
    resnet.load_state_dict(checkpoint['state_dict_resnet'])

    resnet.eval()

    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    with torch.no_grad():
        for i, bs_pair in enumerate(zip(test_loader, random_loader)):
            batch, batch_rand = bs_pair
            X, y = batch
            Xp, _ = batch_rand

            X = X.cuda()
            y = y.cuda()
            Xp = Xp.cuda()

            _, feature = resnet(X)
            logit_compose = feature
        
            test_robust_acc += (logit_compose.max(1)[1] == y).sum().item()
            test_robust_loss += loss.item() * y.size(0)
            test_n += y.size(0)

            # if i > 4 and args.fast:
            #     break

    train_time = time.time()

    resnet.eval()

    # test_loss = 0
    # test_acc = 0
    # test_robust_loss = 0
    test_vanilla_acc = 0
    test_vanilla_loss = 0
    test_n_v = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            X, y = batch

            X = X.cuda()
            y = y.cuda()

            _, logit_compose = resnet(X)
            # TODO: x_pair
            bs_m = feature.size(0)
            j = 0


            test_vanilla_acc += (logit_compose.max(1)[1] == y).sum().item()
            test_vanilla_loss += loss.item() * y.size(0)
            test_n_v += y.size(0)

            # if i > 4 and args.fast:
            #     break
    print("test domain", test_d, " train acc",
          "test vanilla", test_vanilla_acc / test_n_v, "test fd", test_robust_acc / test_n, "\n\n\n")



if __name__ == "__main__":
    main()