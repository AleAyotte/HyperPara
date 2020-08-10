from Experimentation.Cifar10_wide_resnet.Model.WideResNet import WideResNet
from Experimentation.Cifar10_wide_resnet.Dataset.DataManager import load_cifar10, validation_split
from Experimentation.Cifar10_wide_resnet.Training.Training import Trainer
from Optimizer.Domain import ContinuousDomain, DiscreteDomain
import numpy as np
from Scheduler.Scheduler import tune_objective
import torch


# We create the objective function
def objective(hparams, device="cuda:0"):

    # Get the training and validation set and split randomly the validation to create the test set
    trainset, validset, _ = load_cifar10(valid_size=0.15, valid_aug=False)
    validset, testset = validation_split(validset, valid_size=0.3)

    #################################################
    # Wide_ResNet 22-1
    #################################################

    # Set up the hyperparameters
    mixup = [hparams['mixup_0'], hparams['mixup_1'], hparams['mixup_2'], hparams['mixup_3']]
    t_0 = int(hparams['t_0']) if hparams["t_0"] <= 100 else 100

    # Create the model
    widenet = WideResNet(nb_layer=22, widen_factor=1,
                         mixup=mixup, drop_rate=hparams['drop_rate'],
                         task="CIFAR100").to(device)

    # Create the training object
    trainer = Trainer(trainset, validset, device=device, save_path="checkpoint.pth", tol=0.01)

    # Use to determine which algorithm will be used to compute the forward. (That accelerate the compute)
    torch.backends.cudnn.benchmark = True

    # Train the model
    trainer.fit(widenet,
                learning_rate=10**hparams['lr'], momentum=hparams['mom'],
                l2=10**hparams['l2'], batch_size=hparams['b_size'],
                num_epoch=100, warm_up_epoch=0, t_0=t_0,
                eta_min=1e-5, grad_clip=0,
                device=device, mode="Mixup",
                retrain=False, verbose=False)

    # Compute the precision error on test set
    score = trainer.score(testset)

    return 1 - score


# Define the hyperparameter space
h_space = {"lr": ContinuousDomain(-7, -1),
           "l2": ContinuousDomain(-7, -1),
           "mom": ContinuousDomain(0.1, 0.9),
           "drop_rate": ContinuousDomain(-0.1, 0.5),
           "b_size": DiscreteDomain(np.arange(50, 300, 10).tolist()),
           "mixup_0": ContinuousDomain(-0.5, 3.0),
           "mixup_1": ContinuousDomain(-0.5, 3.0),
           "mixup_2": ContinuousDomain(-0.5, 3.0),
           "mixup_3": ContinuousDomain(-0.5, 3.0),
           "t_0": DiscreteDomain(np.arange(10, 120, 1).tolist())}

optim_list = ["GP", "GP", "tpe"]
acq_list = ["EI", "MPI"]
device_list = ["cuda:0", "cuda:1"]
num_iters = 50


def run_experiment():
    tune_objective(objective_func=objective,
                   h_space=h_space,
                   optim_list=optim_list,
                   acq_func_list=acq_list,
                   num_iters=num_iters,
                   device_list=device_list,
                   save_path="Result/Cifar10/",
                   save_each_iter=True,
                   verbose=True)

