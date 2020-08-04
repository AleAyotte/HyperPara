#
# Modèle de base pour «mpi4py».
# Par : Daniel-Junior Dubé
#
# Exemple d'exécution du programme :
#   - `mpiexec -n 4 python main.py` : (Recommendé) Défini par le standard MPI.
#   - `mpirun -n 4 python comm.py` : Commande alternative fournie par plusieurs implémentation d'MPI.
# Documentation : https://mpi4py.readthedocs.io/en/stable/tutorial.html
# Autre ressource sur mpi4py : https://rabernat.github.io/research_computing/parallel-programming-with-mpi-for-python.html

from mpi4py import MPI
import numpy as np
from Manager import HpManager
from Manager.HpManager import HPtype, Hyperparameter, ContinuousDomain, DiscreteDomain
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from tqdm import tqdm
from util import objective, create_msg_task, create_msg_result


if __name__ == '__main__':
    # Objet représentatnt l'espace de calcul.
    comm = MPI.COMM_WORLD
    # Taille de l'espace de travail. Représente le nombre de entités de travail (gestionnaires/travailleurs).
    size = comm.Get_size()
	# Rang du processus actuel. Permet d'identifier le processus de façon unique.
    rank = comm.Get_rank()
    #print("rank : ", rank)

    if rank == 0:
        print("Manager")
        h_space = {"lr": ContinuousDomain(-7, -1),
    			   "alpha": ContinuousDomain(-7, -1),
				   "num_iters": DiscreteDomain([50, 100, 150, 200]),
				   "b_size": DiscreteDomain(np.arange(20, 80, 10).tolist())
				   }
        opt1 = HpManager.get_optimizer(h_space, algo="rand")
        opt2 = HpManager.get_optimizer(h_space, algo="GP", acquisition_fct="MPI")

        sample_x, sample_y = [], []
        nb_random_search = 5

        envoie, recu = (0, 0)
        # Optimizer aleatoire
        while(envoie < nb_random_search):
            #envoie tache
            for tag in range(1, size):
                task = create_msg_task("Next HP rand",
                                       optim=opt1)
                comm.send(task, dest=tag)
                envoie += 1
                if envoie == nb_random_search:
                    break

            # reception tache
            for tag in range(1, size):
                result = comm.recv(source=tag)
                sample_x.extend([result['hparams']])
                sample_y.extend([[result['score']]])
                recu += 1
                if recu == nb_random_search:
                    break

        ###################
        # Optimizer Gaussian process
        ###################
        nb_gauss_search = 15
        envoie, recu = (0, 0)
        pending_x = None
        while (envoie < nb_gauss_search):
            # envoie tache
            for tag in range(1, size):
                task = create_msg_task("Next HP GAUSS",
                                       optim=opt2,
                                       sample_x=sample_x,
                                       sample_y=sample_y,
                                       pending_x=pending_x)
                comm.send(task, dest=tag)
                result = comm.recv(source=tag)
                if result['code'] == 'next HP':
                    if pending_x == None:
                        pending_x = [result['hparams']]
                    else:
                        pending_x.extend(result['hparams'])
                else:
                    print("GROS PROBLEME : synchronisations du canal de communication")
                envoie += 1
                if envoie == nb_gauss_search:
                    break

            # reception resultat avec score
            for tag in range(1, size):
                result = comm.recv(source=tag)
                if result['code'] == "HP + res":
                    sample_x.extend([result['hparams']])
                    sample_y.extend([[result['score']]])
                else:
                    print("Gros PRobleme encore dans les synchronisations du canal de communication")
                recu += 1
                if recu == nb_gauss_search:
                    break

        print("Fin manager")
        ####################
        # Liberer les worker
        ####################
        for tag in range(1, size):
            task = create_msg_task("Fin")
            comm.send(task, dest=tag)

        print("sample_x : ", sample_x)
        print("sample_y : ", sample_y)

    else:
        ############
        # Worker
        ############
        while(True):
            task = comm.recv(source=0)
            # Calcul du prochain
            if task['code'] == "Fin":
                break
            elif task['code'] == "Next HP rand":
                hparams = task['optim'].get_next_hparams()
                score = objective(hparams)
                result = create_msg_result("HP + res",
                                           hparams,
                                           score=score)
                comm.send(result, dest=0)
            elif task['code'] == "Next HP GAUSS":
                hparams = task['optim'].get_next_hparams(task['sample_x'], task['sample_y'], pending_x=None)
                result = create_msg_result("next HP", hparams)
                # Envoie du prochain HP qui sera ajoute aux pending_x dans le manager
                comm.send(result, dest=0)
                # Le worker continue le calcul de fit et de score dans objective
                score = objective(hparams)
                result = create_msg_result("HP + res", hparams, score=score)
                comm.send(result, dest=0)

        print("Fin du travailleur : ", rank)

