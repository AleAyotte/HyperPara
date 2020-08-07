import numpy as np
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
from util import objective
from ordonance import Ordonanceur
from tqdm import tqdm

h_space = {"lr": ContinuousDomain(-7, -1),
           "alpha": ContinuousDomain(-7, -1),
           "num_iters": DiscreteDomain([50, 100, 150, 200]),
           "b_size": DiscreteDomain(np.arange(20, 80, 10).tolist())
           }

nb_random_search, nb_iter = 5, 20

Ordonance = Ordonanceur(h_space, objective, nb_rand_search=5, algos=["tpe", "GP_MPI"])

for it in tqdm(range(20)):
    opt = Ordonance.get_next_optimizer()
    sample_x, sample_y, pending_x = Ordonance.get_sample()
    hparams = opt.get_next_hparams(sample_x, sample_y, pending_x=None)
    Ordonance.append_x(hparams)
    Ordonance.append_y(Ordonance.objective(hparams))



