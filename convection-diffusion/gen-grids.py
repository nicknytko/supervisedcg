import argparse
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize
import pickle
import sys
import time
import pyamg
import bz2
import itertools
import multiprocessing
import functools
import os.path as path

sys.path.append('../lib')
import helpers

from mg import *

parser = argparse.ArgumentParser(description='Randomly generates 2D C/F split grids.  Takes various reference grids and generates random permutations by flipping points at various probabilities')
parser.add_argument('--gridsize', metavar='N', type=int, nargs=1, default=25, help='Number of nodes on the grid', required=False)
parser.add_argument('--iterations', type=int, default=500, help='Number of permutations for each probability', required=False)
parser.add_argument('--input', type=str, help='Matrix to perturb', required=True)
parser.add_argument('--maxthreads', type=int, default=16, help='Maximum number of worker processes to use when generating perturbations', required=False)
parser.add_argument('--randseed', type=int, default=None, help='Random seed to use when perturbing grids', required=False)

args = vars(parser.parse_args())
I = args['iterations']
matname = args['input']
if matname.lower()[-4:] == '.mat':
    matname = matname[:-4]
maxthreads = max(1, args['maxthreads'])

A, b = helpers.load_recirc_flow(path.join('matrices', matname + '.mat'))
t_start = time.time()

print(f'Randomly generating permuted grids for system {matname}')

# grid generation

coarsenings = [ref_all_fine, ref_all_coarse, ref_amg(theta=0.50), ref_coarsen_by_bfs(2), ref_coarsen_by_bfs(3), ref_coarsen_by_bfs(4), ref_coarsen_by_bfs(5)]
#coarsenings = [ref_coarsen_by_bfs(10)]
p_trials = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75])
trial_seeds = None

Ns = []
grids = []
convs = []

def run_trial(cidx):
    grids = []
    convs = []
    S = pyamg.strength.classical_strength_of_connection(A, theta=0.25)
    N = A.shape[0]
    n = int(np.round(A.shape[0] ** 0.5)) - 1
    np.random.seed(trial_seeds[cidx])

    coarsening = coarsenings[cidx]
    name, G = coarsening(A)
    for p in p_trials:
        for i in range(I):
            def permute_grid():
                # Automatically try again for degenerate cases
                while True:
                    perm_G = G.copy()
                    for j in range(N):
                        if np.random.rand(1) < p:
                            perm_G[j] = not perm_G[j]
                    if not np.all(perm_G == False):
                        P = create_interp(A, perm_G)
                        break

                return perm_G, P

            # Create the randomly permuted "perm_C"
            perm_G, P = permute_grid()
            conv = mgv(P, A, omega=0.66)
            if conv > 1:
                conv = 1
            grids.append((perm_G * 2) - 1)
            convs.append(np.round(conv,2))

    print(f'Finished {name}')
    return grids, convs


if __name__ == '__main__':
    grids = []
    convs = []

    # Generate a random seed using the "master seed" that gets distributed to each
    # runner
    np.random.seed(args['randseed'])
    int_max = np.iinfo(np.int32).max
    trial_seeds = (np.random.random_sample(len(coarsenings))*int_max).astype(np.int32)

    threads = min(maxthreads, len(coarsenings))
    with multiprocessing.Pool(processes=threads) as pool:
        outs = pool.map(run_trial, range(len(coarsenings)))
        grids = functools.reduce(lambda acc, v: acc + v[0], outs, [])
        convs = functools.reduce(lambda acc, v: acc + v[1], outs, [])
        for i, out in enumerate(outs):
            coarsening = coarsenings[i]
            name, G = coarsening(A)
        print('finished trials')

    outputs = [
        {
            'var': grids,
            'fname': path.join('splittings', matname + '-cf.pkl.bz2')
        },
        {
            'var': convs,
            'fname': path.join('splittings', matname + '-conv.pkl.bz2')
        }
    ]

    for o in outputs:
        var = o['var']
        fname = o['fname']
        try:
            existing = helpers.pickle_load(fname)
            existing = existing + var
        except Exception as e:
            existing = var
            helpers.pickle_save_bz2(fname, existing)

    t_end = time.time()
    print(f'finished in {int(t_end-t_start)} seconds')
