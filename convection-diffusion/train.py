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

sys.path.append('../lib')
sys.path.append('.')
import helpers

from model import *

parser = argparse.ArgumentParser(description='Trains a GNN to predict some metric on an input grid.')
parser.add_argument('--iterations', type=int, default=-1, help='Number of training iterations to perform', required=False)
parser.add_argument('--batchsize', type=int, default=100, help='Number of entries in each minibatch', required=False)
parser.add_argument('--testsplit', type=float, default=0.85, help='Percent of entries to keep in the training set (as opposed to the testing set).  Should be a value between 0 and 1.', required=False)
parser.add_argument('--matrices', type=str, required=True, nargs='+', help='Matrices to train on.')

cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training on device "{cuda}"')

args = vars(parser.parse_args())

ds = MeshDataset(args['matrices'])
p = args['testsplit']
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']

mse_loss_train = []
mse_loss_test = []
l1_loss_train = []
l1_loss_test = []

def compute_grid_loss(gnn, ds):
    mse = 0
    l1 = 0

    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    num_batches = np.ceil(len(ds) / args['batchsize'])

    with torch.no_grad():
        batches = tg.data.DataLoader(ds, batch_size=args['batchsize'], shuffle=False)
        for batch in batches:
            batch = batch.to(cuda)
            output = gnn(batch).reshape(-1)
            mse += mse_loss(output, batch.y)
            l1 += l1_loss(output, batch.y)

    return mse.cpu()/num_batches, l1.cpu()/num_batches

# compute initial loss
gnn = GNN(cuda).to(cuda)
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
sgd = optim.Adam(gnn.parameters(), lr=0.01)

num_batches = np.ceil(len(train) / args["batchsize"])
print(f'Total {len(ds)} training samples, batch size {args["batchsize"]}, {num_batches} batches.')

mse_initial_train, l1_initial_train = compute_grid_loss(gnn, train)
mse_initial_test, l1_initial_test = compute_grid_loss(gnn, test)
mse_loss_train.append(mse_initial_train); l1_loss_train.append(l1_initial_train)
mse_loss_test.append(mse_initial_test); l1_loss_test.append(l1_initial_test)

print(f'\t e=0 \t MSE Loss: {mse_initial_train:.8f}, L1 Loss: {l1_initial_train:.8f}')

e = 0
while True and iterations != 0:
    batches = tg.data.DataLoader(train, batch_size=args['batchsize'], shuffle=True)
    for i, batch in enumerate(batches):
        batch = batch.to(cuda)
        def compute_loss():
            cur_batch_mse = 0
            output = gnn(batch).reshape(-1)
            cur_batch_mse += mse_loss(output, batch.y)
            cur_batch_mse.backward()
            return cur_batch_mse

        def closure():
            sgd.zero_grad()
            return compute_loss()

        sgd.step(closure)

    mse_epoch_train, l1_epoch_train = compute_grid_loss(gnn, train)
    mse_epoch_test, l1_epoch_test = compute_grid_loss(gnn, test)

    mse_loss_train.append(mse_epoch_train); l1_loss_train.append(l1_epoch_train)
    mse_loss_test.append(mse_epoch_test); l1_loss_test.append(l1_epoch_test)

    print(f'\t e={e+1} \t MSE Loss: {mse_epoch_train:.8f}, L1 Loss: {l1_epoch_train:.8f}')
    e += 1
    if iterations == -1:
        s = input(' (c)ontinue, (s)top [c]:')
        if len(s) > 0 and s.lower()[0] == 's':
            break
    elif e >= iterations:
        break

mse_loss_train = np.array(mse_loss_train); l1_loss_train = np.array(l1_loss_train)
mse_loss_test = np.array(mse_loss_test); l1_loss_test = np.array(l1_loss_test)

torch.save(gnn.state_dict(), f'model/mpnn');
helpers.pickle_save(f'model/mpnn_train_mse.pkl', mse_loss_train); print('saved training mse loss')
helpers.pickle_save(f'model/mpnn_test_mse.pkl', mse_loss_test); print('saved testing mse loss')
helpers.pickle_save(f'model/mpnn_train_l1.pkl', l1_loss_train); print('saved training l1 loss')
helpers.pickle_save(f'model/mpnn_test_l1.pkl', l1_loss_test); print('saved testing l1 loss')
