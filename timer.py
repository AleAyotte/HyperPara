import os
import time

import time
start_time = time.time()
#os.system('python exemple_ordonanceur.py')
os.system('mpiexec -n 2 python test_parallelisation.py')
print("--- %s seconds ---" % (time.time() - start_time))
