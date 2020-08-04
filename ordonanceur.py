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
from util import objective




"""h_space = {"lr": ContinuousDomain(-7, -1),
           "alpha": ContinuousDomain(-7, -1),
           "num_iters": DiscreteDomain([50, 100, 150, 200]),
           "b_size": DiscreteDomain(np.arange(20, 80, 10).tolist())
           }"""

"""####################################
#               TPE
####################################
print("TPE OPTIMIZATION")
print("h_space : ", h_space)
opt1 = HpManager.get_optimizer(h_space, algo="rand")
opt2 = HpManager.get_optimizer(h_space, algo="tpe")

sample_x, sample_y = [], []

for it in tqdm(range(20)):

    if it < 5:
        hparams = opt1.get_next_hparams()
    else:
        # For Gabriel
        # pending_x are the point that currently evaluate in another process.
        hparams = opt2.get_next_hparams(sample_x, sample_y, pending_x=None)
        print(hparams)

    sample_x.extend([hparams])
    sample_y.extend([[objective(hparams)]])

print(sample_y)

best_idx = np.argmin(sample_y)
print("\nThe best hyperparameters is {}.\n\nFor a score of {}\n\n".format(
    sample_x[best_idx],
    sample_y[best_idx]
))"""


if __name__ == '__main__':
	# Objet représentatnt l'espace de calcul.
	comm = MPI.COMM_WORLD

	# Taille de l'espace de travail. Représente le nombre de entités de travail (gestionnaires/travailleurs).
	size = comm.Get_size()
	# print(size)
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
		opt2 = HpManager.get_optimizer(h_space, algo="tpe")

		sample_x, sample_y = [], []
		nb_ranfom_search = 5
		envoie, recu = (0, 0)
		a = [0,1,2,3,4]
		b = [0,0,0,0,0]
		while(envoie < nb_ranfom_search):
			#envoie tache
			for tag in range(1, size):
				#data_envoie = "message envoye du manager rank: {0}, au travailleur rank: {1}".format(rank, tag)
				task = {
  					"code": "Next HP",
  					"optim": opt1,
				}
				comm.send(task, dest=tag)
				envoie += 1
				if envoie == nb_ranfom_search:
					break

			# reception tache
			for tag in range(1, size):
				result = comm.recv(source=tag)
				sample_x.extend([result['X']])
				sample_y.extend([[result['Y']]])
				recu += 1
				if recu == nb_ranfom_search:
					break
			print("un tour de while")
		print("fin while manager")
		print("sample_x : ", sample_x)
		print("sample_y : ", sample_y)
		"""
			for tag in range(1, size):
				data_recu = comm.recv(source=tag)
				recu += 1
				if recu == nb_ranfom_search:
					break
		"""
		"""for tag in range(1, size):
			data = "message envoye du manager rank: {0}, au travailleur rank: {1}".format(rank, tag)
			comm.send(data, dest=tag)
"""
	else:
		print("Worker")
		while(True):
			task = comm.recv(source= 0)
			if task['code'] == "Next HP":
				hparams = task['optim'].get_next_hparams()
			#sample_x.extend([hparams])
			score = objective(hparams)
			result = {
  					"X": hparams,
  					"Y": score,
				}
			comm.send(result, dest=0)
			#print("rank du travailleur : ", rank, " data recu : ", data)

"""
if rank == 0:
	# ...
	# Gestionnaire
	# ...
	# Lecture des matrices dans des arrays numpy N * N
	reference_job = divideTask(nWorkers, nJobs)
	while 
	for i in range(nWorkers):
		data = {
			'ref': reference_job[i],
			'A': A,
			'B': B,
			'N': N
		}
		tag = i+1
		comm.send(data, dest=tag)

	C = np.zeros((N, N))    # contient le resultat
	for i in range(nWorkers):
		r = i+1
		res = comm.recv(source= r)
		begin_task_code = res['ref'][0]
		end_task_code = res['ref'][1]
		result = res['res']
		N = data['N']
		# On les range dans la matrice C
		for cij in range( begin_task_code, end_task_code + 1 ):
			i, j = indexes(cij, N)
			C[i, j] = result[cij - res['ref'][0]]

	print("Fin de l'algorithme")
	print("La matrice C : ")
	print(C)

else :
	# ...
	# Travailleur
	# ...
	# Réception de données par appel bloquant :

	# data['ref'] : list contenant des ensemble de reference aux cellules a traiter
	# data['A'] : matrice A
	# data['B'] : matrice B
	# N : taille des matrices
	data = comm.recv(source= 0)

	result = np.zeros(data['ref'][1] - data['ref'][0] + 1)

	for cij in range(data['ref'][0], data['ref'][1]+1):
		i, j = indexes(cij, data['N'])
		result[cij - data['ref'][0]] = np.sum(data['A'][i] * data['B'][:, j])

	res = {
		'ref': data['ref'],
		'res': result
	}
	comm.send(res, dest=0 )
	print("Fin du processus ", rank)
"""


