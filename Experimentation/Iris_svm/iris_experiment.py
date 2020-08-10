from Experimentation.Iris_svm.classifiers.svm import SVMClassifier
import numpy as np
from Optimizer.Domain import ContinuousDomain
from Scheduler.Scheduler import tune_objective


def create_objective_func(dataset):
    def objective_func(hparams, device="cpu"):

        hyperparameter = {
            'C': 10**hparams['C'],
            'gamma': 10**hparams['gamma']
        }

        # Five step of cross validation to reduce the noise that affect the loss function.
        result = []
        for _ in range(5):
            net = SVMClassifier(hyperparameter, dataset)

            net.train()
            result.append(net.evaluate("Testing"))

        return np.mean(result)
    return objective_func


def run_experiment(setting=1):
    """

    :param setting: A integer that indicate which set of optimizer will be used in the experimentation
    """
    objective = create_objective_func("iris")

    h_space = {"C": ContinuousDomain(-8, 0), "gamma": ContinuousDomain(-8, 0)}

    if setting == 1:
        optim_list = ["GP", "GP", "tpe"]
        acq_list = ["EI", "MPI"]
        path = "Result/Iris/Setting1"
    else:
        optim_list = ["tpe"]
        acq_list = None
        path = "Result/Iris/Setting2"

    num_iters = 250

    tune_objective(objective_func=objective,
                   h_space=h_space,
                   optim_list=optim_list,
                   acq_func_list=acq_list,
                   num_iters=num_iters,
                   save_path=path,
                   save_each_iter=False,
                   verbose=False)
