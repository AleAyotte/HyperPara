"""
    @file:              HpManager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/07/2020

    @Reference:         1) https://github.com/AleAyotte/AutoML-MAT523
    @Description:       This file provide a class that standardize and translate the search of the different library.
                        This file is highly inspired on another project Ref1 that has been done by the current author.
"""

from enum import Enum, unique
from hyperopt import hp


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