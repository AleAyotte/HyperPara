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
from util import objective, create_msg_result, create_msg_task

if __name__ == '__main__':
	# Objet représentatnt l'espace de calcul.
	comm = MPI.COMM_WORLD
	# Taille de l'espace de travail. Représente le nombre de entités de travail (gestionnaires/travailleurs).
	size = comm.Get_size()
	# Rang du processus actuel. Permet d'identifier le processus de façon unique.
	rank = comm.Get_rank()
	#print("rank : ", rank)

	if rank == 0:
		h_space = {
			"lr": ContinuousDomain(-7, -1),
			"alpha": ContinuousDomain(-7, -1),
			"num_iters": DiscreteDomain([50, 100, 150, 200]),
			"b_size": DiscreteDomain(np.arange(20, 80, 10).tolist())
		}
		opt1 = HpManager.get_optimizer(h_space, algo="rand")
		opt2 = HpManager.get_optimizer(h_space, algo="tpe")

		sample_x, sample_y = [], []
		nb_random_search = 5

		envoie, recu = (0, 0)
		# Optimizer aleatoire
		while(envoie < nb_random_search):
			#envoie tache
			for tag in range(1, size):
				task = create_msg_task("Next HP", optim=opt1)
				comm.send(task, dest=tag)
				envoie += 1
				if envoie == nb_random_search:
					break

			# reception resultat
			for tag in range(1, size):
				result = comm.recv(source=tag)
				sample_x.extend([result['hparams']])
				sample_y.extend([[result['score']]])
				recu += 1
				if recu == nb_random_search:
					break

		###################
		# Optimizer TPE
		###################
		nb_tpe_search = 15
		envoie, recu = (0, 0)
		while (envoie < nb_tpe_search):
			# envoie des taches
			for tag in range(1, size):
				task = create_msg_task("Next HP TPE", optim=opt2, sample_x=sample_x, sample_y=sample_y)
				comm.send(task, dest=tag)
				envoie += 1
				if envoie == nb_tpe_search:
					break

			# reception des resultats
			for tag in range(1, size):
				result = comm.recv(source=tag)
				if result['code'] == "HP + score":
					sample_x.extend([result['hparams']])
					sample_y.extend([[result['score']]])
				recu += 1
				if recu == nb_tpe_search:
					break
		# On libere les workers
		for tag in range(1, size):
			task = create_msg_task("Fin")
			comm.send(task, dest=tag)
		print("sample_x : ", sample_x)
		print("sample_y : ", sample_y)

	else:
		########
		# Worker
		########
		while(True):
			task = comm.recv(source= 0)
			if task['code'] == "Fin":
				break
			elif task['code'] == "Next HP":
				hparams = task['optim'].get_next_hparams()
			elif task['code'] == "Next HP TPE":
				hparams = task['optim'].get_next_hparams(task['sample_x'], task['sample_y'], pending_x=None)
			score = objective(hparams)
			result = create_msg_result("HP + score", hparams=hparams, score=score)
			comm.send(result, dest=0)
