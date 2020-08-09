"""
    @file:              HpManager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/07/2020

    @Reference:         1) https://github.com/AleAyotte/AutoML-MAT523
    @Description:       This file provide a class that standardize and translate the search of the different library.
                        This file is highly inspired on another project Ref1 that has been done by the current author.
"""
from copy import deepcopy
from Optimizer.Domain import ContinuousDomain, DiscreteDomain


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
