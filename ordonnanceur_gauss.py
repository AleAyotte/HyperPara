#
# Modèle de base pour «mpi4py».
# Par : ...
#
# Exemple d'exécution du programme :
#   - `mpiexec -n 4 python ordonnanceur_gauss.py` : (Recommendé) Défini par le standard MPI.
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
from util import objective, create_msg_task, create_msg_result, argument_parser
import argparse


if __name__ == '__main__':
    # Objet représentatnt l'espace de calcul.
    comm = MPI.COMM_WORLD
    # Taille de l'espace de travail. Représente le nombre de entités de travail (gestionnaires/travailleurs).
    size = comm.Get_size()
	# Rang du processus actuel. Permet d'identifier le processus de façon unique.
    rank = comm.Get_rank()

    if rank == 0:
        args = argument_parser()
        if args.nb_iter < args.nb_rand:
            raise Exception("Erreur dans l'argument nb_iter : doit être >= nb_rand")

        h_space = {"lr": ContinuousDomain(-7, -1),
    			   "alpha": ContinuousDomain(-7, -1),
				   "num_iters": DiscreteDomain([50, 100, 150, 200]),
				   "b_size": DiscreteDomain(np.arange(20, 80, 10).tolist())
				   }
        ###################
        # Optimizer Gaussian process
        ###################
        # Le veritable nombre de random search peut etre plus grand que celui-ci car le choix de faire une recherche
        # aleatoire ou Gaussienne depend de la taille de sampl_x et la somme de sample_x et pending_x
        nb_random_search, nb_iter = args.nb_rand, args.nb_iter

        if args.algo == "GP":
            opt = HpManager.get_optimizer(h_space, n_rand_point=nb_random_search, algo="GP", acquisition_fct="MPI")
        elif args.algo == "tpe":
            opt = HpManager.get_optimizer(h_space, n_rand_point=5, algo="tpe")
        else:
            raise Exception("Erreur dans l'argument algo : tpe ou GP sont possible")

        envoie, recu = (0, 0)
        sample_x, sample_y = [], []
        pending_x = None

        while (envoie < nb_iter):
            # envoie tache
            for tag in range(1, size):
                task = create_msg_task("Next HP GAUSS",
                                       optim=opt,
                                       sample_x=sample_x,
                                       sample_y=sample_y,
                                       pending_x=pending_x)
                # Envoie tache de chercher le prochain Hparams
                comm.send(task, dest=tag)
                # Recoi le prochain vecteur d'hparams qui sera calculé par le travailleur tag
                result = comm.recv(source=tag)
                if result['code'] == 'next HP':
                    if pending_x == None:
                        pending_x = [result['hparams']]
                    else:
                        pending_x.extend(result['hparams'])
                else:
                    print("GROS PROBLEME : synchronisations du canal de communication")
                envoie += 1

                if envoie == nb_iter or envoie == nb_random_search:
                    break

            # reception resultat
            for tag in range(1, size):
                # Reception du hparams et du score obtenue
                result = comm.recv(source=tag)
                if result['code'] == "HP + res":
                    sample_x.extend([result['hparams']])
                    sample_y.extend([[result['score']]])
                else:
                    print("Gros Probleme encore dans les synchronisations du canal de communication")
                recu += 1
                if recu == nb_iter or envoie == nb_random_search:
                    break

        ##########################
        # Liberer les travailleurs
        ##########################
        for tag in range(1, size):
            task = create_msg_task("Fin")
            comm.send(task, dest=tag)
        ##########################

        ####################
        # Affichage Resultat
        ####################
        print("sample_x : ", sample_x)
        print("sample_y : ", sample_y, "\n")
        best_idx = np.argmin(sample_y)
        print("\nFor the algorithm {}, the best hyperparameters is {}.\n\nFor a score of {}\n\n".format(
            args.algo,
            sample_x[best_idx],
            sample_y[best_idx]
        ))

    else:
        ##############
        # Travailleurs
        ##############
        while(True):
            task = comm.recv(source=0)
            if task['code'] == "Fin":
                # Il n'y a plus de tache a accomplir, le processus du travailleur peut terminer
                break
            elif task['code'] == "Next HP GAUSS":
                # Calcul du prochain HP
                hparams = task['optim'].get_next_hparams(task['sample_x'], task['sample_y'], pending_x=None)
                result = create_msg_result("next HP", hparams)
                # Envoie du prochain HP qui sera ajoute aux pending_x dans le manager
                comm.send(result, dest=0)
                # Le worker continue le calcul de fit et de score dans objective
                score = objective(hparams)
                result = create_msg_result("HP + res", hparams, score=score)
                comm.send(result, dest=0)


