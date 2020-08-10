"""
    @file:              Scheduler.py
    @Author:            Alexandre Ayotte
    @Creation Date:     09/08/2020

    @Description:       This provide the code to dispatch the optimization of a given objective function.
"""

from Scheduler import Manager, Worker
from mpi4py import MPI
from time import time
from tqdm import tqdm


def tune_objective(objective_func, h_space, optim_list, num_iters, acq_func_list=None, num_init_rand=5,
                   device_list=None, save_path="", save_each_iter=False, verbose=False):
    """
    Tune an objective function

    :param objective_func: The objective function to evaluate
    :param h_space: A dictionary of Domain object that represent the hyperparameters space
    :param optim_list: A list of optimizer algorithm to instantiate
    :param num_iters: Number of time the objective function will be evaluated
    :param acq_func_list: A list of acquisition function that will be used the optimizer of type gaussian process
    :param num_init_rand: Number configuration that will be sample with a random optimizer before using the
                              optimizers in the list.
    :param device_list: A list of device on which the objective will be evaluate
    :param save_path: the path to directory where the result will saved. (Default="")
    :param save_each_iter: If true sample_x, sample_y and best_y are save into csv each time that add_to_sample
                           is called. (Default=False)
    :param verbose: If true, then show a progress bar
    """
    # Object that represent the compute space.
    comm = MPI.COMM_WORLD
    num_worker = comm.Get_size() - 1

    # Rank of the current process
    rank = comm.Get_rank()

    # Manager
    if rank == 0:
        manager = Manager.Manager(optim_list=optim_list,
                                  acq_func_list=acq_func_list,
                                  h_space=h_space,
                                  num_init_rand=num_init_rand,
                                  save_path=save_path,
                                  save_each_iter=save_each_iter)

        num_worker_working = 0
        num_job_left = num_iters

        # We start the choronometer
        begin = time()

        # Dispatch the job to the worker process
        for process in range(1, num_worker+1):
            if num_job_left > 0:
                hparams = manager.get_next_point()

                comm.send(hparams, dest=process)
                num_worker_working += 1
                num_job_left -= 1

        with tqdm(total=num_iters, disable=(not verbose)) as t:
            while num_worker_working > 0:
                # The manager collect the result
                message = comm.recv(source=MPI.ANY_SOURCE)
                worker_id, hparams, result = message

                manager.add_to_sample(hparams, result)
                num_worker_working -= 1
                t.update()

                # If their is job left to do, we send another job to the worker
                if num_job_left > 0:
                    hparams = manager.get_next_point()
                    comm.send(hparams, dest=worker_id)
                    num_worker_working += 1
                    num_job_left -= 1

                else:
                    comm.send("STOP", dest=worker_id)

        # We end the choronometer and print the execution time
        end = time()
        print("Execution time: {}".format(end-begin))

        # We save the result
        manager.save_sample()
        
    # Worker
    elif rank != 0:
        device = device_list[rank-1] if device_list is not None else None

        # We create the worker object
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
