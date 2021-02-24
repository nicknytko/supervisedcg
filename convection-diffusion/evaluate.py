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

parser = argparse.ArgumentParser(description='Evaluate trained GNN on some dataset.')
parser.add_argument('--matrices', type=str, required=True, nargs='+', help='Matrices to train on.')
parser.add_argument('--batchsize', type=int, default=2000, help='Number of entries in each minibatch', required=False)

args = vars(parser.parse_args())
ds = MeshDataset(args['matrices'])

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Evaluating on device "{cuda}"')

gnn = load_model(cuda)

def eval_dataset(gnn, ds):
    num_batches = np.ceil(len(ds) / args['batchsize'])
    conv_output = np.zeros(len(ds))
    idx_cur = 0

    with torch.no_grad():
        batches = tg.data.DataLoader(ds, batch_size=args['batchsize'], shuffle=False)
        for batch in batches:
            batch = batch.to(cuda)
            output = gnn(batch).reshape(-1).cpu().flatten()
            n = len(output)
            conv_output[idx_cur:idx_cur+n] = output
            idx_cur += n
    return conv_output

conv_ref = ds.convs
conv_pred = eval_dataset(gnn, ds)

print(conv_ref, conv_pred)
mats = '_'.join(sorted(args['matrices']))
helpers.pickle_save(f'model/predictions_{mats}.pkl', np.column_stack((conv_ref, conv_pred)))

plt.figure()
plt.plot(conv_ref, conv_pred, 'o')
plt.xlabel('True Convergence')
plt.ylabel('Predicted Convergence')
plt.show(block=True)
