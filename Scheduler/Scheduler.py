from Scheduler import Manager, Worker
import sys
from mpi4py import MPI
import numpy as np


def tune_objective(objective_func, h_space, optim_list, num_iters, acq_func_list=None,
                   num_init_rand=5, device_list=None):
    """
    Tune an objective function

    :param objective_func: The objective function to evaluate
    :param h_space: A dictionary of Domain object that represent the hyperparameters space
    :param optim_list:
    :param num_iters: Number of time the objective function will be evaluated
    :param acq_func_list:
    :param num_init_rand:
    :param device_list:
    :return:
    """
    # Objet représentatnt l'espace de calcul.
    comm = MPI.COMM_WORLD
    num_worker = comm.Get_size() - 1

    # Rank of the current process
    rank = comm.Get_rank()

    # Manager
    if rank == 0:
        manager = Manager.Manager(optim_list=optim_list,
                                  acq_func_list=acq_func_list,
                                  h_space=h_space,
                                  num_init_rand=num_init_rand)
        num_worker_working = 0
        num_job_left = num_iters

        for process in range(1, num_worker):
            hparams = manager.get_next_point()

            comm.send(hparams, dest=process)
            num_worker_working += 1

        while num_worker_working:
            message = comm.recv(source=MPI.ANY_SOURCE)
            worker_id, hparams, result = message

            manager.add_to_sample(hparams, result)
            num_job_left -= 1
            num_worker_working -= 1

            if num_job_left > 0:
                hparams = manager.get_next_point()
                comm.send(hparams, dest=worker_id)
                num_worker_working += 1

            else:
                comm.send("STOP", dest=worker_id)

        manager.save_sample()
        
    # Worker
    elif rank != 0:
        device = device_list[rank-1] if device_list is not None else None

        # W create the worker
        worker = Worker.Worker(objective=objective_func,
                               device=device,
                               proc_id=rank)
        still_job_to_do = True

        while still_job_to_do:
            message = comm.recv(source=0)

            # If the message is not a string then this is the configuration to evaluate
            if not isinstance(message, str):
                comm.send(worker.evaluate_obj(message), dest=0)
            else:
                still_job_to_do = False
