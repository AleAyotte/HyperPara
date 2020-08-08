from Experimentation.Cifar10_wide_resnet.Model.WideResNet import WideResNet
from Experimentation.Cifar10_wide_resnet.Dataset.DataManager import load_cifar100
from Experimentation.Cifar10_wide_resnet.Training.Training import Trainer
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
import numpy as np
import torch


def objective(hparams, device="cuda:0"):
    trainset, validset, testset = load_cifar100(valid_size=0.10, valid_aug=False)

    #################################################
    # Wide_ResNet 22-1
    #################################################
    mixup = [hparams['mixup_0'], hparams['mixup_1'], hparams['mixup_2'], hparams['mixup_3']]
    t_0 = int(hparams['t_0']) if hparams["t_0"] <= 100 else 100
    widenet = WideResNet(nb_layer=22, widen_factor=1,
                         mixup=mixup, drop_rate=hparams['drop_rate'],
                         task="CIFAR100").to(device)

    trainer = Trainer(trainset, validset, device=device, save_path="checkpoint.pth", tol=0.01)

    torch.backends.cudnn.benchmark = True

    trainer.fit(widenet,
                learning_rate=10**hparams['lr'], momentum=hparams['mom'],
                l2=10**hparams['l2'], batch_size=hparams['b_size'],
                num_epoch=100, warm_up_epoch=0, t_0=t_0,
                eta_min=1e-5, grad_clip=0,
                device=device, mode="Mixup", retrain=False)

    score = trainer.score(testset)

    return 1 - score


h_space = {"lr": ContinuousDomain(-7, -1),
           "l2": ContinuousDomain(-7, -1),
           "mom": ContinuousDomain(0.1, 0.9),
           "drop_rate": ContinuousDomain(-0.1, 0.5),
           "b_size": DiscreteDomain(np.arange(50, 150, 10).tolist()),
           "mixup_0": ContinuousDomain(-0.5, 3.0),
           "mixup_1": ContinuousDomain(-0.5, 3.0),
           "mixup_2": ContinuousDomain(-0.5, 3.0),
           "mixup_3": ContinuousDomain(-0.5, 3.0),
           "t_0": DiscreteDomain(np.arange(10, 150, 1).tolist())}


