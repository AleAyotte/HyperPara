"""
    @file:              HpManager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/07/2020

    @Reference:         1) https://github.com/AleAyotte/AutoML-MAT523
    @Description:       This file provide a class that standardize and translate the search of the different library.
                        This file is highly inspired on another project Ref1 that has been done by the current author.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum, unique
from GPyOpt.methods import BayesianOptimization
from hyperopt import hp, fmin, rand, tpe
import numpy as np


@unique
class HPtype(Enum):
    """
    Class containing possible types of hyper-parameters

    From @Ref1
    """

    real = 1
    integer = 2
    categorical = 3


class Hyperparameter:
    """
    From @Ref1
    """
    def __init__(self, name, _type, value=None):
        """
        Class that defines an hyper-parameter

        :param name: Name of the hyper-parameter
        :param _type: One type out of HPtype (real,.integer, categorical)
        :param value: List with the value of the hyper-parameter
        """

        self.name = name
        self.type = _type
        self.value = value


class HPOptimizer(ABC):
    """
    Optimizer abstract class.
    """
    def __init__(self, hp_space, keys):
        """
        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param keys:  A list that contain the name of each hyperparameters.
        """

        super().__init__()
        self.hp_space = hp_space
        self.keys = [key for key in keys]

    @abstractmethod
    def get_next_hparams(self, sample_x, sample_y, pending_x=None):
        pass

    def list_to_dict(self, hparams):
        """
        Convert a list of hparams into a dictionnary of hparams.

        :param hparams: The list of hparams.
        :return: A dictionnary of hyperparameters.
        """

        assert len(hparams) == len(self.keys)
        hp_dict = {}

        for it in range(len(hparams)):
            hp_dict[self.keys[it]] = hparams[it]

        return hp_dict


class HyperoptOptimizer(HPOptimizer):
    """
    The HyperOpt optimizer class.
    """

    def __init__(self, hp_space, algo="tpe"):
        """
        The construction of the GPyOpt optimizer.

        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param algo: The algorithm that will to define the next point to evaluate.
        """

        space = HyperoptSearchSpace(hp_space)

        super().__init__(space, hp_space.keys())

        self.last_hparams = None
        self.algo = tpe.suggest if algo == "tpe" else rand.suggest

        # Hyperopt used ordered dictionary so we need to sort the keys list
        self.keys.sort()

    def build_objective_function(self, sample_x, sample_y):
        """
        Build the objective that hyperopt will try minimise. If the hparams has already been evaluated,
        the objective function will return the corresponding result. If the hparams has never been evaluated,
        the objective function will save the hparams into self.last_hparams and will return 0.

        :param sample_x: A list of list that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :return:An objective function to tune with hp_tuner.
        """

        def obj(hparams):
            assert np.shape(sample_x)[1] == len(hparams.keys())

            hp_list = [hparams[key] for key in hparams.keys()]

            for idx in range(len(sample_x)):
                if np.all(sample_x[idx] == hp_list):
                    return sample_y[idx][0]
            else:
                self.last_hparams = hparams
                return 0

        return obj

    def get_next_hparams(self, sample_x=None, sample_y=None, pending_x=None):
        """
        This function suggest the next list hyperparameters to evaluate according a given sample
        of evaluated list of hyperparameters.

        :param sample_x: A list of list that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :param pending_x: A list of list that define all hyperparameters that are evaluating
                          right now by another process.
        :return: The next list of hyperparameters to evaluate.
        """

        obj_func = self.build_objective_function(sample_x, sample_y)
        sample_dict = [self.list_to_dict(hparams) for hparams in sample_x]

        _ = fmin(
            fn=obj_func,
            space=self.hp_space.space,
            algo=self.algo,
            points_to_evaluate=sample_dict,
            verbose=False,
            show_progressbar=False,
            max_evals=1
        )

        return self.last_hparams


class GpyOptOptimizer(HPOptimizer):
    """
    The GPyOpt optimizer class.
    """

    def __init__(self, hp_space, algo="GP", acquisition_fct="EI"):
        """
        The construction of the GPyOpt optimizer.

        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param algo: The algorithm that will be used to define the surrogate model
        :param acquisition_fct: The function that will be used to define the next point to evaluate
        """
        space = GPyOptSearchSpace(hp_space)

        super().__init__(space, hp_space.keys())
        self.model = algo

        self.acq_fct = acquisition_fct + "_MCMC" if algo == "GP_MCMC" else acquisition_fct

    def get_next_hparams(self, sample_x=None, sample_y=None, pending_x=None):
        """
        This function suggest the next list hyperparameters to evaluate according a given sample
        of evaluated list of hyperparameters.

        :param sample_x: A list of list that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :param pending_x: A list of list that define all hyperparameters that are evaluating
                          right now by another process.
        :return: The next list of hyperparameters to evaluate.
        """

        # We define the surrogate model
        bo = BayesianOptimization(
            f=None,
            model_type=self.model,
            acquisition_type=self.acq_fct,
            domain=list(self.hp_space.space.values()),
            X=sample_x, Y=sample_y,
            de_duplication=True  # required to consider the pending hparams.
        )

        hp_list = bo.suggest_next_locations(pending_X=pending_x)
        return self.list_to_dict(hp_list[0])


class SearchSpace:
    """
    From @Ref1
    """

    def __init__(self, space):
        """
        Definition of a search space for our hyper-parameters
        """

        self.default_space = space
        self.space = space
        self.log_scaled_hyperparam = []

    def reset(self):
        """
        Resets search space to default
        """

        self.space = deepcopy(self.default_space)
        self.log_scaled_hyperparam.clear()

    def change_hyperparameter_type(self, hyperparam, new_type):
        """
        Changes hyper-parameter type in search space (only useful in GPyOpt search spaces)

        :param hyperparam: Name of the hyperparameter
        :param new_type: Type from HPtype
        """

        pass

    def reformat_for_tuning(self):
        """
        Reformats search space so it is now compatible with hyper-parameter optimization method
        """

        pass

    def save_as_log_scaled(self, hyperparam):
        """
        Saves hyper-parameter's name that is log scaled

        :param hyperparam: Name of the hyperparameter
        """

        self.log_scaled_hyperparam.append(hyperparam)

    def __getitem__(self, key):
        return self.space[key]

    def __setitem__(self, key, value):
        self.space[key] = value


class HyperoptSearchSpace(SearchSpace):
    """
    From @Ref1
    """
    def __init__(self, hp_space):
        """
        Class that defines a compatible search space with Hyperopt package hyper-parameter optimization algorithm

        :param hp_space: A dictionary that represent the hyperparameters space.
        """

        space = {}

        for hparam_name in hp_space.keys():
            space[hparam_name] = hp_space[hparam_name].compatible_format('tpe', hparam_name)

        super(HyperoptSearchSpace, self).__init__(space)

    def reformat_for_tuning(self):
        """
        Inserts the whole built space in a hp.choice object that can now be pass as a space parameter
        in Hyperopt hyper-parameter optimization algorithm
        """

        for hyperparam in list(self.space.keys()):

            # We check if the hyper-parameter space is an hyperOpt object (if yes, the user wants it to be tune)
            if type(self[hyperparam]).__name__ == 'Hyperparameter':
                self.space.pop(hyperparam)

        self.space = hp.choice('space', [self.space])


class GPyOptSearchSpace(SearchSpace):
    """
    From @Ref1
    """
    def __init__(self, hp_space):
        """
        Class that defines a compatible search space with GPyOpt package hyper-parameter optimization algorithm

        :param hp_space: A dictionary that represent the hyperparameters space.
        """

        space = {}
        self.categorical_vars = {}

        for hparam_name in hp_space.keys():
            hparam = hp_space[hparam_name]

            if isinstance(hparam, ContinuousDomain):
                _type = 'continuous'
            else:
                _type = 'discrete'

            space[hparam_name] = {'name': hparam_name,
                                  'type': 'continuous',
                                  'domain': hparam.compatible_format('gaussian_process', hparam_name),
                                  'dimensionality': 1}

        super(GPyOptSearchSpace, self).__init__(space)
        self.hyperparameters_to_tune = None


@unique
class DomainType(Enum):
    """
    Class containing possible types of hyper-parameters

    From @Ref1
    """

    continuous = 1
    discrete = 3


class Domain:
    """
    From @Ref1
    """
    def __init__(self, type_):
        """
        Abstract (parent) class that represents a domain for hyper-parameter's possible values

        :param type_: One type of domain among DomainType
        """

        self.type = type_


class ContinuousDomain(Domain):
    """
    From @Ref1
    """
    def __init__(self, lower_bound, upper_bound, log_scaled=False):
        """
        Class that generates a continuous domain

        :param lower_bound: Lowest possible value (included)
        :param upper_bound: Highest possible value (included)
        :param log_scaled: If True, hyper-parameter will now be seen as 10^x where x follows a uniform(lb,ub)
        """

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        self.lb = lower_bound
        self.ub = upper_bound
        self.log_scaled = log_scaled

        super(ContinuousDomain, self).__init__(DomainType.continuous)

    def compatible_format(self, tuner_method, label):
        """
        Builds the correct format of a uniform distribution according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Uniform distribution compatible with method used by HPtuner
        """

        if tuner_method in ['random_search', 'tpe', 'annealing']:
            return hp.uniform(label, self.lb, self.ub)

        elif tuner_method == 'gaussian_process':
            return tuple([self.lb, self.ub])


class DiscreteDomain(Domain):
    """
    Took from @Ref1
    """
    def __init__(self, possible_values):
        """
        Class that generates a domain with possible discrete values of an hyper-parameter

        :param possible_values: list of values
        """

        self.values = possible_values

        super(DiscreteDomain, self).__init__(DomainType.discrete)

    def compatible_format(self, tuner_method, label):
        """
        Builds the correct format of discrete set of values according to the method used by the tuner

        :param tuner_method: Name of the method employed by the HPtuner.
        :param label: String defining the name of the hyper-parameter
        :return: Set of values compatible with method used by HPtuner
        """

        if tuner_method == 'grid_search':
            return self.values

        elif tuner_method in ['random_search', 'tpe', 'annealing']:
            return hp.choice(label, self.values)

        elif tuner_method == 'gaussian_process':
            return tuple(self.values)
