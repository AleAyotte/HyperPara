"""
    @file:              Manager.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/08/2020

    @Description:       This file provide a class that will be used to create a manager that will compute with a list
                        of optimizer the next configuration to be evaluate with the objective function by a worker.

"""
# from Manager.HpManager import get_optimizer
from Optimizer.Optimizer import get_optimizer
import csv


class Manager:
    """

    """
    def __init__(self, optim_list, acq_func_list, h_space, num_init_rand):
        """
        The Manager constructor

        :param optim_list: A list of optimizer algorithm to instantiate
        :param acq_func_list: A list of acquisition function that will be used the optimizer of type gaussian process
        :param h_space: The hyperparameters space that will be used by each optimizer to construct the surrogate model.
        :param num_init_rand: Number configuration that will be sample with a random optimizer before using the
                              optimizers in the list.
        """

        self.optimizers = []
        self.sample_x = []
        self.sample_y = []
        self.best_y = []
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

    def get_next_point(self):
        """
        Get the next configuration to evaluate by sampling with the next optimizer to used.

        :return: A dictionary that represent the next configuration to evaluate.
        """

        optimizer = self.optimizers[self.next_optim]
        next_x = optimizer.get_next_hparams(sample_x=self.sample_x,
                                            sample_y=self.sample_y,
                                            pending_x=self.pending_x)
        self.pending_x.append(next_x)
        self.next_optim = (self.next_optim + 1) % self.num_optim

        return next_x

    def add_to_sample(self, config, result):
        """
        Add a configuration $lambda$ and a result into the sample and remove the configuration from the pending list.

        :param config: A dictionary that represent the configuration that have evaluate with the objective function.
        :param result: A float that represent the result of evaluation.
        """

        self.sample_x.append(config)
        self.sample_y.append([result])
        self.pending_x.remove(config)

        if len(self.best_y) > 0:
            self.best_y.append(min(result, self.best_y[-1]))
        else:
            self.best_y.append(result)

    def save_sample(self):
        """
        Save sample_x, sample_y, and best_y into separated csv file.
        """

        csv_column = list(self.sample_x[0].keys())
        csv_column.append("result")

        # We save the results of iteration in a csv file.
        with open('sample_y.csv', 'w') as f:
            for res in self.sample_y:
                f.write("%s\n" % res[0])

        # We save the best result of each iteration in a csv file.
        with open('best_y.csv', 'w') as f:
            for res in self.best_y:
                f.write("%s\n" % res)

        # We save the hyperparameters and their corresponding results of each iteration in a csv file.
        with open('sample_x.csv', 'w') as f:
            for it in range(len(csv_column)):
                if it == len(csv_column) - 1:
                    f.write("%s\n" % csv_column[it])
                else:
                    f.write("%s," % csv_column[it])

            for it1 in range(len(self.sample_x)):
                hparams = self.sample_x[it1]
                keys = list(hparams.keys())

                for it2 in range(len(keys)):
                    key = keys[it2]
                    if it2 == len(keys) - 1:
                        f.write("%s, %s\n" % (hparams[key], self.sample_y[it1][0]))
                    else:
                        f.write("%s," % hparams[key])
