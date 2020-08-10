from Experimentation.Breast_cancer_mlp.classifiers.fully_connected import FullyConnectedClassifier
import numpy as np
from Optimizer.Domain import ContinuousDomain, DiscreteDomain
from Scheduler.Scheduler import tune_objective


def create_objective_func(dataset):

    def objective_func(hparams, device="cpu"):
        hidden_layer_size = []

        for _ in range(int(hparams["num_layer"])):
            hidden_layer_size.append(int(hparams["layer_size"]))

        hyperparameter = {
            'hidden_layer_sizes': tuple(hidden_layer_size),
            'learning_rate_init': 10**hparams['lr'],
            'alpha': 10**hparams['alpha'],
            'max_iter': 500,
            'batch_size': int(hparams['b_size'])
        }

        # Five step of cross validation to reduce the noise that affect the loss function.
        result = []
        for _ in range(5):
            net = FullyConnectedClassifier(hyperparameter, dataset)

            net.train()

            result.append(net.evaluate("Testing"))
        return np.mean(result)
    return objective_func


def run_experiment():

    objective = create_objective_func("breast_cancer")

    h_space = {
        "lr": ContinuousDomain(-7, -1),
        "alpha": ContinuousDomain(-7, -1),
        "b_size": DiscreteDomain(np.arange(50, 500, 10).tolist()),
        "num_layer": DiscreteDomain(np.arange(1, 20, 1).tolist()),
        "layer_size": DiscreteDomain(np.arange(20, 100, 1).tolist()),
    }

    optim_list = ["GP", "GP", "tpe"]
    acq_list = ["EI", "MPI"]
    num_iters = 250

    tune_objective(objective_func=objective,
                   h_space=h_space,
                   optim_list=optim_list,
                   acq_func_list=acq_list,
                   num_iters=num_iters,
                   save_path="Result/BreastCancer/",
                   save_each_iter=False,
                   verbose=False)
