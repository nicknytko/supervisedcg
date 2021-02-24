import sys
import os
import subprocess
import time

arguments = sys.argv[1:]
matrices = list(map(lambda l: l[:-4], os.listdir('matrices')))

t_start = time.time()

for mat in matrices:
    print(f' -- Generating for {mat} -- ')
    subprocess.call(['python3', 'gen-grids.py'] + arguments + ['--input', mat])

t_end = time.time()

print(f' -- Finished generating all matrices in {t_end-t_start} seconds -- ')
