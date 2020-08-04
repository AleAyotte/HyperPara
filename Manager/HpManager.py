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
from functools import partial

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


def get_optimizer(hp_space, algo, n_rand_point=10, acquisition_fct="EI"):
    """

    :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
    :param algo: The algorithm that will to define the next point to evaluate.
    :param n_rand_point: Number of points that will sample randomly before starting the optimization.
    :param acquisition_fct: The function that will be used to define the next point to evaluate.
                            Only used by Gaussian process.
    :return:
    """
    if algo == "tpe" or algo == "random" or algo == "rand":
        return HyperoptOptimizer(hp_space, n_rand_point=n_rand_point, algo=algo)
    elif algo == "GP" or algo == "GP_MCMC":
        return GpyOptOptimizer(hp_space, n_rand_point=n_rand_point, algo=algo, acquisition_fct=acquisition_fct)


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
        self.keys.sort()

    @abstractmethod
    def get_next_hparams(self, sample_x, sample_y, pending_x=None):
        pass

    def _list_to_dict(self, hparams):
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

    def __init__(self, hp_space, n_rand_point, algo="tpe"):
        """
        The construction of the GPyOpt optimizer.

        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param n_rand_point: Number of points that will sample randomly before starting the optimization.
        :param algo: The algorithm that will to define the next point to evaluate.
        """

        space = HyperoptSearchSpace(hp_space)

        super().__init__(space, hp_space.keys())

        self.last_hparams = None
        self.algo = partial(tpe.suggest, n_startup_jobs=n_rand_point) if algo == "tpe" else rand.suggest

    def build_objective_function(self, sample_x, sample_y):
        """
        Build the objective that hyperopt will try minimise. If the hparams has already been evaluated,
        the objective function will return the corresponding result. If the hparams has never been evaluated,
        the objective function will save the hparams into self.last_hparams and will return 0.

        :param sample_x: A list of dictionary that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :return:An objective function to tune with hp_tuner.
        """

        if sample_x is not None and len(sample_x) > 0:
            def obj(hparams):
                assert len(sample_x[0].keys()) == len(hparams.keys())

                hp_list = [hparams[key] for key in hparams.keys()]
                for idx in range(len(sample_x)):
                    _dict = sample_x[idx]
                    sample = [_dict[key] for key in _dict.keys()]

                    if np.all(sample == hp_list):
                        return sample_y[idx][0]

                else:
                    self.last_hparams = hparams
                    return 0
        else:
            def obj(hparams):
                self.last_hparams = hparams
                return 0
        return obj

    def reformat_point_to_evaluate(self, sample_x):
        """
        The values of the discrete dimensions, by their index in the possible values list.
        Exemple: if "batch_size": DiscreteDomain([50, 100, 150, 200]).
        then sample = [{"batch_size": 100}, {"batch_size": 50}] -> sample = [{"batch_size": 1}, {"batch_size": 0}]

        :param sample_x: A list of dictionary that define all tested hyperparameters.
        :return: A list of dictionary that define all tested hyperparameters but compatible with hyperopt.
        """

        disc_space = self.hp_space.discrete_space
        sample = deepcopy(sample_x)

        for it in range(len(sample)):
            _dict = sample[it]
            for key in disc_space.keys():
                _dict[key] = self.hp_space.get_discrete_index(key, _dict[key])
        return sample

    def get_next_hparams(self, sample_x=None, sample_y=None, pending_x=None):
        """
        This function suggest the next list hyperparameters to evaluate according a given sample
        of evaluated list of hyperparameters.

        :param sample_x: A list of dictionary that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :param pending_x: A list of list that define all hyperparameters that are evaluating
                          right now by another process.
        :return: The next list of hyperparameters to evaluate.
        """

        obj_func = self.build_objective_function(sample_x, sample_y)

        if sample_x is not None:
            sample = self.reformat_point_to_evaluate(sample_x)
        else:
            sample = sample_x

        _ = fmin(
            fn=obj_func,
            space=self.hp_space.space,
            algo=self.algo,
            points_to_evaluate=sample,
            verbose=False,
            show_progressbar=False,
            max_evals=1
        )

        return self.last_hparams


class GpyOptOptimizer(HPOptimizer):
    """
    The GPyOpt optimizer class.
    """

    def __init__(self, hp_space, n_rand_point, algo="GP", acquisition_fct="EI"):
        """
        The construction of the GPyOpt optimizer.

        :param hp_space: A dictionary that contain object of type continuous domain and discrete domain.
        :param n_rand_point: Number of points that will sample randomly before starting the optimization.
        :param algo: The algorithm that will be used to define the surrogate model
        :param acquisition_fct: The function that will be used to define the next point to evaluate
        """

        space = GPyOptSearchSpace(hp_space)

        super().__init__(space, hp_space.keys())

        # We save it for random search.
        self.rand_space = hp_space
        self.model = algo
        self.initial_random_point = n_rand_point
        self.acq_fct = acquisition_fct + "_MCMC" if algo == "GP_MCMC" else acquisition_fct

    def get_next_hparams(self, sample_x=None, sample_y=None, pending_x=None):
        """
        This function suggest the next list hyperparameters to evaluate according a given sample
        of evaluated list of hyperparameters.

        :param sample_x: A list of dictionary that define all tested hyperparameters.
        :param sample_y: A list that give the score of each list of tested hyperparameters
        :param pending_x: A list of dictionary that define all hyperparameters that are evaluating
                          right now by another process.
        :return: The next list of hyperparameters to evaluate.
        """

        if len(sample_x) < self.initial_random_point:
            random_opt = get_optimizer(self.rand_space, "rand")
            return random_opt.get_next_hparams()

        else:
            sample = np.array([[_dict[key] for key in self.keys] for _dict in sample_x])

            if pending_x is not None:
                pending_x = np.array([[_dict[key] for key in self.keys] for _dict in pending_x])

            # We define the surrogate model
            bo = BayesianOptimization(
                f=None,
                model_type=self.model,
                acquisition_type=self.acq_fct,
                domain=list(self.hp_space.space.values()),
                X=sample, Y=sample_y,
                de_duplication=True  # required to consider the pending hparams.
            )

            hp_list = bo.suggest_next_locations(pending_X=pending_x)
            return self._list_to_dict(hp_list[0])


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
        self.discrete_space = {}

        for hparam_name in sorted(hp_space.keys()):
            space[hparam_name] = hp_space[hparam_name].compatible_format('tpe', hparam_name)

            if isinstance(hp_space[hparam_name], DiscreteDomain):
                self.discrete_space[hparam_name] = hp_space[hparam_name].values

        super(HyperoptSearchSpace, self).__init__(space)

    def get_discrete_index(self, hparam_names, value):
        """
        Return the index list of a value in the possible values list according to the name of the hyperparameter.

        :param hparam_names: A string that represent the hyperparameter name.
        :param value: The value. Must be present in the possible values list
        :return: The index list of the value in the list.
        """

        return self.discrete_space[hparam_names].index(value)


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

        for hparam_name in sorted(hp_space.keys()):
            hparam = hp_space[hparam_name]

            if isinstance(hparam, ContinuousDomain):
                _type = 'continuous'
            else:
                _type = 'discrete'

            space[hparam_name] = {'name': hparam_name,
                                  'type': _type,
                                  'domain': hparam.compatible_format('gaussian_process', hparam_name),
                                  'dimensionality': 1}

        super(GPyOptSearchSpace, self).__init__(space)


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
    def __init__(self, lower_bound, upper_bound):
        """
        Class that generates a continuous domain

        :param lower_bound: Lowest possible value (included)
        :param upper_bound: Highest possible value (included)
        """

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        self.lb = lower_bound
        self.ub = upper_bound

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
