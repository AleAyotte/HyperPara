import os
import sys
from typing import List, Any, Union

import numpy as np
from mpi4py import MPI
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
import time
import random
from util import objective
from ordonance import Ordonanceur

WORKTAG = 0
DIETAG = 1
READYTAG = 2
NEXT_HPTAG = 3
RESULTTAG = 4

#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

def manager(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    h_space = {"lr": ContinuousDomain(-7, -1),
               "alpha": ContinuousDomain(-7, -1),
               "num_iters": DiscreteDomain([50, 100, 150, 200]),
               "b_size": DiscreteDomain(np.arange(20, 80, 1).tolist())
               }

    nb_random_search, nb_iter = 5, 20

    Ord = Ordonanceur(h_space, objective, nb_rand_search=nb_random_search, algos=["tpe", "GP_MPI"])

    """ Manager """
    status = MPI.Status()
    nb_search = 0
    while(True):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        source = status.Get_source()
        if tag == READYTAG:
            data = random.randrange(2, 5)
            x, y, pending = Ord.get_sample()
            task = {
                "opti": Ord.get_next_optimizer(),
                "sample_x": x,
                "sample_y": y,
                "pending_x": pending,
                'objective': Ord.get_objective_function()
            }
            comm.send(task, dest=source, tag=WORKTAG)
        elif tag == NEXT_HPTAG:
            # Manager recoit le prochain HP qui sera bientot calcule
            Ord.append_pending(data)
        elif tag == RESULTTAG:
            # Le manager recoi un resultat dans data Il doit ajouter les nouvelle donnee
            Ord.remove_pending(data[0])
            Ord.append_x(data[0])
            Ord.append_y(data[1])
            nb_search += 1
        if nb_search == nb_iter:
            break

    for i in range(1, size):
        # Enoie le meassage de s'arreter a tous les processus Worker
        comm.send(None, dest=i, tag=DIETAG)
    Ord.show_result()

def worker(comm):
    """ Worker """
    rank = comm.Get_rank()
    status = MPI.Status()
    while(True):
        comm.send(None, dest=0, tag=READYTAG)
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        if tag == DIETAG:
            break
        elif tag == WORKTAG:
            hparams = task['opti'].get_next_hparams(task['sample_x'], task['sample_y'], pending_x=task['pending_x'])
            comm.send(hparams, dest=0, tag=NEXT_HPTAG)
            score = task['objective'](hparams)
            comm.send([hparams, score], dest=0, tag=RESULTTAG)


def main():
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    my_name = MPI.Get_processor_name()

    if my_rank == 0:
        manager(comm)
    else:
        worker(comm)


if __name__ == '__main__':
    main()