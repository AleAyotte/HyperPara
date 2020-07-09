"""
    @file:              HpManager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/07/2020

    @Reference:         1) https://github.com/AleAyotte/AutoML-MAT523
    @Description:       This file provide a class that standardize and translate the search of the different library.
                        This file is highly inspired on another project Ref1 that has been done by the current author.
"""

from copy import deepcopy
from enum import Enum, unique
from GPyOpt.methods import BayesianOptimization
from hyperopt import hp, fmin, rand, tpe


@unique
class HPtype(Enum):

    """
    Class containing possible types of hyper-parameters
    """

    real = 1
    integer = 2
    categorical = 3


class Hyperparameter:

    def __init__(self, name, type, value=None):

        """
        Class that defines an hyper-parameter

        :param name: Name of the hyper-parameter
        :param type: One type out of HPtype (real,.integer, categorical)
        :param value: List with the value of the hyper-parameter
        """

        self.name = name
        self.type = type
        self.value = value


class SearchSpace:

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

    def __init__(self, model):

        """
        Class that defines a compatible search space with Hyperopt package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}

        for hyperparam in model.HP_space:
            space[hyperparam] = model.HP_space[hyperparam]

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

    def __init__(self, model):

        """
        Class that defines a compatible search space with GPyOpt package hyper-parameter optimization algorithm

        :param model: Available model from Model.py
        """

        space = {}
        self.categorical_vars = {}

        for hyperparam in model.HP_space:

            hp_initial_value = model.HP_space[hyperparam].value[0]

            if model.HP_space[hyperparam].type.value == HPtype.categorical.value:

                space[hyperparam] = {'name': hyperparam, 'type': 'categorical',
                                     'domain': (hp_initial_value,), 'dimensionality': 1}

                self.categorical_vars[hyperparam] = {}

            else:
                space[hyperparam] = {'name': hyperparam, 'type': 'discrete',
                                     'domain': (hp_initial_value,), 'dimensionality': 1}

        super(GPyOptSearchSpace, self).__init__(space)
        self.hyperparameters_to_tune = None

    def change_hyperparameter_type(self, hp_to_fix, new_type):

        """
        Changes hyper-parameter type in the search space

        :param hp_to_fix: Name of the hyper-parameter which we want to change his type
        :param new_type: The new type (one among DomainType)
        """
        self[hp_to_fix]['type'] = new_type.name

    def reformat_for_tuning(self):

        """
        Converts the dictionnary to a list containing only internal dictionaries.
        Only keep hyper-parameters that has more than a unique discrete value as a domain
        """

        for hyperparam in list(self.space.keys()):

            # We save the length of the domain (must be at least 2)
            domain_length = len(self[hyperparam]['domain'])

            # If there's no search to be done with the hyper-parameter we do not consider it anymore in the tuning
            if domain_length == 1:
                self.space.pop(hyperparam)

            # If the hyper-parameter is categorical, we change strings for integer.
            elif self[hyperparam]['type'] == 'categorical':

                # We save the possible values of the categorical variables in forms of strings and also integers
                choices = list(self[hyperparam]['domain'])
                integer_encoding = tuple(range(domain_length))

                # We change the domain of our space for the tuple with all values (int) possible
                self[hyperparam]['domain'] = integer_encoding

                # We save the choices associated with each integer in our dictionary
                for i in integer_encoding:
                    self.categorical_vars[hyperparam][i] = choices[i]

        self.hyperparameters_to_tune = list(self.space.keys())
        self.space = list(self.space.values())
        print(self.hyperparameters_to_tune)

        if len(self.hyperparameters_to_tune) == 0:
            raise Exception('The search space has not been modified yet. Each hyper-parameter has only a discrete'
                            'domain of length 1 and no tuning can be done yet')

    def change_to_dict(self, hyper_paramater_values):

        """
        Builds a dictionary of hyper-parameters
        :param hyper_paramater_values: 2d numpy array of hyper-parameters' values
        :return: dictionary of hyper-parameters
        """

        # We initialize a dictionary and an index
        hp_dict, i = {}, 0

        # We extract hyper-parameters' values
        hyper_parameter_values = hyper_paramater_values[0]

        # We build the dict and transform back categorical variables
        for hyperparam in self.hyperparameters_to_tune:

            hp_value = hyper_parameter_values[i]  # Represents a dictionary key for categorical vars

            if hyperparam in self.categorical_vars:
                hp_dict[hyperparam] = self.categorical_vars[hyperparam][hp_value]

            else:
                hp_dict[hyperparam] = hp_value

            i += 1

        return hp_dict

    def __setitem__(self, key, value):
        self.space[key]['domain'] = value

@unique
class DomainType(Enum):

    """
    Class containing possible types of hyper-parameters
    """

    continuous = 1
    discrete = 3


class Domain:

    def __init__(self, type):

        """
        Abstract (parent) class that represents a domain for hyper-parameter's possible values

        :param type: One type of domain among DomainType
        """

        self.type = type


class ContinuousDomain(Domain):

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
