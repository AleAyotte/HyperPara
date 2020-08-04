import numpy as np
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from tqdm import tqdm


def load_breast_cancer_dataset(scaled=True, test_split=0.2):

    """
    Loads the breast cancer wisconsin dataset for classication task

    :param scaled: True for scaling the data.
    :param test_split: test_split: Proportion of the dataset that will be use as test data (Default 0.2 = 20%).
    :return: 4 numpy arrays for training features, training labels, testing features and testing labels respectively.
    """

    data, target = load_breast_cancer(True)

    if scaled:
        data = preprocessing.scale(data)

    x_train, x_test, t_train, t_test = train_test_split(data, target, test_size=test_split)
    return x_train, t_train, x_test, t_test


def objective(hparams, device="cpu"):
    x_train, t_train, x_test, t_test = load_breast_cancer_dataset()
    net = MLPClassifier(
        hidden_layer_sizes=(50, 10, 2),
        learning_rate_init=10**hparams['lr'],
        alpha=10**hparams['alpha'],
        max_iter=int(hparams['num_iters']),
        batch_size=int(hparams['b_size'])
    )

    net.fit(x_train, t_train)

    return 1 - net.score(x_test, t_test)


h_space = {"lr": ContinuousDomain(-7, -1),
           "alpha": ContinuousDomain(-7, -1),
           "num_iters": DiscreteDomain([50, 100, 150, 200]),
           "b_size": DiscreteDomain(np.arange(20, 80, 1).tolist())
           }

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
print("\nThe best hyperparameters is {}.\n\nFor a score of {}\n\n".format(
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
print("\nThe best hyperparameters is {}.\n\nFor a score of {}\n\n".format(
    sample_x[best_idx],
    sample_y[best_idx]
))