"""
    @file:              Manager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/08/2020

    @Description:       This file provide a class that will be used to create a worker that will evaluate
                        the cost function on a given vector of hyperparameters and return the information
                        that need to be send to the Manager.
"""
from Manager.HpManager import get_optimizer


class Manager:
    """

    """
    def __init__(self, optim_list, acq_func_list, h_space, num_init_rand):
        """
        The manager class constructor
        """

        self.optimizers = []
        self.sample_x = []
        self.sample_y = []
        self.pending_x = []
        self.next_optim = 0
        self.num_optim = len(optim_list)

        self.__create_optimizers(optim_list, acq_func_list, h_space, num_init_rand)

    def __create_optimizers(self, optim_list, acq_func_list, h_space, num_init_rand):
        """
        Create the list of optimizer that will be used to determine the next configuration to evaluate by the worker.

        :param optim_list: A list of optimizer algorithm to instantiate
        :param acq_func_list: A list of acquisition function that will be used the optimizer of type gaussian process
        :param h_space: The hyperparameters space that will be used by each optimizer to construct the surrogate model.
        :param num_init_rand: Number configuration that will be sample with a random optimizer before using the
                              optimizers in the list.
        """
        for algo_name in optim_list:

            acq_func = None
            if algo_name[0:2] == "GP":
                assert len(acq_func_list) > 0, "Every Gaussien Process need an acquisition"
                acq_func = acq_func_list.pop(0)

            self.optimizers.append(get_optimizer(hp_space=h_space,
                                                 algo=algo_name,
                                                 n_rand_point=num_init_rand,
                                                 acquisition_fct=acq_func)
                                   )

