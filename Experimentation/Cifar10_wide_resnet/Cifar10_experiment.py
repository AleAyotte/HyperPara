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
    t_0 = int(hparams['t_0']) if hparams["t_0"] <= 150 else 150

    # Create the model
    widenet = WideResNet(nb_layer=22, widen_factor=3,
                         mixup=mixup, drop_rate=hparams['drop_rate'],
                         ).to(device)

    checkpoint = "checkpoint_" + device[-1] + ".pth"
    # Create the training object
    trainer = Trainer(trainset, validset, device=device, save_path=checkpoint, tol=0.01)

    # Use to determine which algorithm will be used to compute the forward. (That accelerate the compute)
    torch.backends.cudnn.benchmark = True

    # Train the model
    trainer.fit(widenet,
                learning_rate=10**hparams['lr'], momentum=hparams['mom'],
                l2=10**hparams['l2'], batch_size=int(hparams['b_size']),
                num_epoch=150, warm_up_epoch=5, t_0=int(t_0),
                eta_min=1e-6, grad_clip=0,
                device=device, mode="Mixup",
                retrain=False, verbose=False)

    # Compute the precision error on test set
    score = trainer.score(testset)

    return 1 - score


# Define the hyperparameter space
h_space = {"lr": ContinuousDomain(-7, 1),
           "l2": ContinuousDomain(-7, -1),
           "mom": ContinuousDomain(0.1, 0.9),
           "drop_rate": ContinuousDomain(-0.1, 0.5),
           "b_size": DiscreteDomain(np.arange(50, 300, 10).tolist()),
           "mixup_0": ContinuousDomain(-0.1, 3.0),
           "mixup_1": ContinuousDomain(-0.1, 3.0),
           "mixup_2": ContinuousDomain(-0.1, 3.0),
           "mixup_3": ContinuousDomain(-0.1, 3.0),
           "t_0": DiscreteDomain(np.arange(10, 160, 1).tolist())}


def run_experiment(setting=1):

    if setting == 1:
        optim_list = ["GP", "GP", "tpe"]
        acq_list = ["EI", "MPI"]
        path = "Result/Cifar10/Setting1/"
    elif setting == 2:
        optim_list = ["tpe"]
        acq_list = None
        path = "Result/Cifar10/Setting2/"
    else:
        optim_list = ["GP"]
        acq_list = ["EI"]
        path = "Result/Cifar10/Setting3/"

    device_list = ["cuda:0", "cuda:1"]
    num_iters = 50

    tune_objective(objective_func=objective,
                   h_space=h_space,
                   optim_list=optim_list,
                   acq_func_list=acq_list,
                   num_iters=num_iters,
                   device_list=device_list,
                   save_path=path,
                   save_each_iter=True,
                   verbose=True)

