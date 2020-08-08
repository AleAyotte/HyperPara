from classifiers.svm import SVMClassifier
import numpy as np
from Manager import HpManager
from Manager.HpManager import ContinuousDomain
from tqdm import tqdm
import argparse


def create_objective_func(dataset):
    def objective_func(hparams, device="cpu"):

        hyperparameter = {
            'C': hparams['C'],
            'gamma': hparams['gamma']
        }

        net = SVMClassifier(hyperparameter, dataset)

        net.train()

        return 1 - net.evaluate("Testing")
    return objective_func


if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--dataset', action='store', type=str, required=True)
    args = my_parser.parse_args()

    objective = create_objective_func(args.dataset)

    h_space = {"C": ContinuousDomain(1, 3), "gamma": ContinuousDomain(0.5, 2)}

    ####################################
    #               TPE
    ####################################
    print("TPE OPTIMIZATION")
    opt1 = HpManager.get_optimizer(h_space, n_rand_point=5, algo="tpe")

    sample_x, sample_y = [], []

    for it in tqdm(range(20)):
        # For Gabriel
        # pending_x are the point that currently evaluate in another process.
        hparams = opt1.get_next_hparams(sample_x, sample_y, pending_x=None)

        sample_x.extend([hparams])
        sample_y.extend([[objective(hparams)]])

    print(sample_y)

    best_idx = np.argmin(sample_y)

    print("\nThe best hyperparameters is {}.\n\n For a score of {}\n\n".format(
        sample_x[best_idx],
        sample_y[best_idx]
    ))

    ####################################
    #       GAUSSIAN PROCESS
    ####################################

    print("GAUSSIAN PROCESS OPTIMIZATION")

    # Algorithm can be GP or GP_MCMC
    # Acquisition function can be: EI, MPI, LCB
    opt2 = HpManager.get_optimizer(h_space, n_rand_point=5, algo="GP", acquisition_fct="MPI")

    sample_x, sample_y = [], []

    for it in tqdm(range(20)):
        # For Gabriel
        # pending_x are the point that currently evaluate in another process.
        hparams = opt2.get_next_hparams(sample_x, sample_y, pending_x=None)

        sample_x.extend([hparams])
        sample_y.extend([[objective(hparams)]])

    print(sample_y)

    best_idx = np.argmin(sample_y)

    print("\nThe best hyperparameters is {}.\n\n For a score of {}\n\n".format(
        sample_x[best_idx],
        sample_y[best_idx]
    ))