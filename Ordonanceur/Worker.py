"""
    @file:              Worker.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/08/2020

    @Description:       This file provide a class that will be used to create a worker that will evaluate
                        the cost function on a given vector of hyperparameters and return the information
                        that need to be send to the Manager.
"""


class Worker:
    """

    """
    def __init__(self, objective, device, proc_id):
        """
        The worker class

        :param objective: The objective function to evaluate.
        :param device: The device on which the objective function will be evaluate.
        :param proc_id: The id of the current process.
        """
        self.objective = objective
        self.device = device
        self.id = proc_id

    def evaluate_obj(self, hparams):
        """
        Evaluate the objective function on a given hyperparameters vector and return a list of
        information that will be used by the Manager

        :param hparams: A dictionary of hyperparameters to evaluate with the objective function
        :return: A list that contain (1) the current process id, (2) the dictionary of hyperparameters
                 and (3) result of the evaluation ($y = L(\lambda)$).
        """

        return [self.id, hparams, self.objective(hparams, self.device)]
