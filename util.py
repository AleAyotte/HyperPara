import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing


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

def create_msg_task(code, optim=None, sample_x=None, sample_y=None, pending_x=None):
    task_dict = {
        "code": code,
        "optim": optim,
        "sample_x": sample_x,
        "sample_y": sample_y,
        "pending_x": pending_x,
    }
    return task_dict

def create_msg_result(code, hparams, score=None):
    result_dict = {
        "code": code,
        "hparams": hparams,
        "score": score,
    }
    return result_dict

