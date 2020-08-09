from Experimentation.Iris_svm.classifiers.svm import SVMClassifier
from Optimizer.Domain import ContinuousDomain
from Scheduler.Scheduler import tune_objective


def create_objective_func(dataset):
    def objective_func(hparams, device="cpu"):

        hyperparameter = {
            'C': 10**hparams['C'],
            'gamma': 10**hparams['gamma']
        }

        net = SVMClassifier(hyperparameter, dataset)

        net.train()

        return 1 - net.evaluate("Testing")
    return objective_func


def run_experiment():

    objective = create_objective_func("iris")

    h_space = {"C": ContinuousDomain(-8, 0), "gamma": ContinuousDomain(-8, 0)}

    optim_list = ["GP", "GP", "tpe"]
    acq_list = ["EI", "MPI"]
    num_iters = 250

    tune_objective(objective_func=objective,
                   h_space=h_space,
                   optim_list=optim_list,
                   acq_func_list=acq_list,
                   num_iters=num_iters,
                   save_path="Result/Iris/",
                   save_each_iter=False)
