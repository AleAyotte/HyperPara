import numpy as np
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from util import objective, create_msg_task, create_msg_result, argument_parser
import argparse
from copy import deepcopy
import random

class Ordonanceur:
    """
    Organizing class to make an hyper-Parameter search
    """

    def __init__(self, hp_space, objective_function, nb_rand_search=5, sample_x=None, sample_y=None, algos=None):
        """
        The construction of the GPyOpt optimizer.

        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param n_rand_point: Number of points that will sample randomly before starting the optimization.
        :param algo: The algorithm that will to define the next point to evaluate.
        """

        # space = HyperoptSearchSpace(hp_space)   # ajouter au optimizer

        # super().__init__(space, hp_space.keys())
        self.objective = objective_function

        if algos is None:
            self.optimizers = \
                [HpManager.get_optimizer(hp_space, n_rand_point=nb_rand_search, algo="GP", acquisition_fct="MPI"),
                 HpManager.get_optimizer(hp_space, n_rand_point=nb_rand_search, algo="tpe")]
        else:
            self.optimizers = []
            for alg in algos:
                self.optimizers.append(self.get_optimizer(alg, hp_space, nb_rand=nb_rand_search))

        self.sample_x = [] if sample_x is None else sample_x
        self.sample_y = [] if sample_y is None else sample_y
        self.pending_x = []

    @staticmethod
    def get_optimizer(algo_str, space, nb_rand=5):
        if algo_str == "tpe":
            return HpManager.get_optimizer(space, n_rand_point=nb_rand, algo="tpe")
        elif algo_str == "GP_MPI":
            return HpManager.get_optimizer(space, n_rand_point=nb_rand, algo="GP", acquisition_fct="MPI")

    def get_next_optimizer(self):
        """
        return randomly an optimizer from list of optimizers
        """
        i = random.randrange(len(self.optimizers))
        return self.optimizers[i]

    def get_sample(self):
        return deepcopy(self.sample_x), deepcopy(self.sample_y), deepcopy(self.pending_x)

    def train_and_Get_score(self, hparams):
        return self.objective(hparams)

    def get_objective_function(self):
        return self.objective

    def append_x(self, hp):
        self.sample_x.extend([hp])

    def append_y(self, score):
        self.sample_y.extend([[score]])

    def append_pending(self, pending):
        self.pending_x.append(pending)

    def remove_pending(self, pending):
        self.pending_x.remove(pending)

    def show_result(self):
        best_idx = np.argmin(self.sample_y)
        print("\nThe best hyperparameters is {}.\n\nFor a score of {}\n\n".format(
            self.sample_x[best_idx],
            self.sample_y[best_idx]
        ))