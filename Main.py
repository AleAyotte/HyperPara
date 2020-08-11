"""
    @file:              Main.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/08/2020

    @Description:       This file is used to call and start the experimentation
"""

import argparse
from Experimentation.Cifar10_wide_resnet.Cifar10_experiment import run_experiment as run_exp_cifar10
from Experimentation.Breast_cancer_mlp.breast_cancer_experiment import run_experiment as run_exp_bc
from Experimentation.Iris_svm.iris_experiment import run_experiment as run_exp_iris


def argument_parser():
    """
        A parser the get the name of the experiment that we want to do
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default="Iris",
                        choices=["Iris", "BreastCancer", "Cifar10"])
    parser.add_argument('--setting', type=int, default=1,
                        choices=[1, 2, 3])
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()
    exp_name = args.exp

    if exp_name == "Iris":
        run_exp_iris(setting=args.setting)
    elif exp_name == "BreastCancer":
        run_exp_bc(setting=args.setting)
    elif exp_name == "Cifar10":
        run_exp_cifar10()
    else:
        raise Exception("There is no experimentation named {}".format(exp_name))
