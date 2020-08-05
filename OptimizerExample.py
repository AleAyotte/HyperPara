import numpy as np
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from tqdm import tqdm
from Model.classifiers.fully_connected import FullyConnectedClassifier
from Model.classifiers.svm import SVMClassifier

def objective(hparams, device="cpu"):

    # hyperparameter = {
    #     'hidden_layer_sizes': (50, 10, 2),
    #     'learning_rate_init': 10**hparams['learning_rate_init'],
    #     'alpha': 10**hparams['alpha'],
    #     'max_iter': int(hparams['num_iters']),
    #     'batch_size': int(hparams['b_size'])
    # }

    hyperparameter = {
        'C': hparams['C'],
        'gamma': hparams['gamma']
    }

    net = SVMClassifier(hyperparameter)

    net.train()

    return 1 - net.evaluate("Testing")


# h_space = {"lr": ContinuousDomain(-7, -1),
#            "alpha": ContinuousDomain(-7, -1),
#            "num_iters": DiscreteDomain([50, 100, 150, 200]),
#            "b_size": DiscreteDomain(np.arange(20, 80, 1).tolist())
#            }

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