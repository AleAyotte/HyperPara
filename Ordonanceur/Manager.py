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

