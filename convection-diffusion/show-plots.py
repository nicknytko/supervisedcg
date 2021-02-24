import numpy as np
import numpy.linalg as la
import pickle
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import sys
import argparse
import scipy.sparse as sp
import matplotlib.pyplot as plt

sys.path.append('../lib')
sys.path.append('.')
import helpers
from model import *

parser = argparse.ArgumentParser(description='Show plots for dataset(s) that have already been evaluated.')
parser.add_argument('--matrices', type=str, required=True, nargs='+', help='Matrices')

args = vars(parser.parse_args())
mats = '_'.join(sorted(args['matrices']))

preds = np.array(helpers.pickle_load(f'model/predictions_{mats}.pkl'))
print(preds)

n = preds.shape[0]
conv_ref = preds[:,0]
conv_pred = preds[:,1]

print('n', n)
print('MSE', np.sum((conv_pred - conv_ref)**2) / n)

plt.figure()
plt.plot(conv_ref, conv_pred, 'o', label='Predicted Values', markersize=2)
plt.xlabel('True Convergence')
plt.ylabel('Predicted Convergence')
plt.plot([0,1],[0,1], label='Diagonal line')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()
plt.show(block=True)
