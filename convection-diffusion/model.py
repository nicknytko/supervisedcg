import numpy as np
import pickle
import torch
import torch._six
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch_geometric as tg
import sys
from rangedict import RangeDict

sys.path.append('../lib')
sys.path.append('.')
import helpers

from mg import *

if __name__ == '__main__':
    debug = print
else:
    def noop(*args, **kwargs):
        pass
    debug = noop


def scipy_csr_to_pytorch_sparse(A):
    Acoo = A.tocoo()
    indices = np.row_stack([Acoo.row, Acoo.col])
    values = Acoo.data
    shape = Acoo.shape
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


class ShapeDebugger(nn.Module):
    def __init__(self, msg=''):
        super(ShapeDebugger, self).__init__()
        self.msg = msg

    def forward(self, x):
        print(f' {self.msg}: {x.shape}')
        return x


class TensorLambda(nn.Module):
    def __init__(self, func):
        super(TensorLambda, self).__init__()
        self.f = func

    def forward(self, x):
        return self.f(x)


class GNN(nn.Module):
    def __init__(self, device):
        super(GNN, self).__init__()
        self.device = device

        # self.conv1 = tg.nn.GCNConv(1, 2)
        # self.conv2 = tg.nn.GCNConv(2, 3)
        # self.conv3 = tg.nn.GCNConv(3, 2)
        # self.conv4 = tg.nn.GCNConv(2, 1)

        # self.conv1 = tg.nn.NNConv(2, 2, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 4), nn.ReLU()))
        # self.conv2 = tg.nn.NNConv(2, 3, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 6), nn.ReLU()))
        # self.conv3 = tg.nn.NNConv(3, 2, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 6), nn.ReLU()))
        # self.conv4 = tg.nn.NNConv(2, 1, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU()))

        self.conv1 = tg.nn.NNConv(2, 2, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 4), nn.ReLU()))
        self.conv2 = tg.nn.NNConv(2, 3, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 6), nn.ReLU()))
        self.conv3 = tg.nn.NNConv(3, 3, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 9), nn.ReLU()))
        self.conv4 = tg.nn.NNConv(3, 2, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 3), nn.ReLU(), nn.Linear(3, 6), nn.ReLU()))
        self.conv5 = tg.nn.NNConv(2, 1, nn=nn.Sequential(TensorLambda(lambda x: x.reshape(-1, 1).float()), nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU()))

        # self.conv1 = tg.nn.SignedConv(2, 2, first_aggr=True)
        # self.conv2 = tg.nn.SignedConv(2, 3, first_aggr=False)
        # self.conv3 = tg.nn.SignedConv(3, 3, first_aggr=False)
        # self.conv4 = tg.nn.SignedConv(3, 2, first_aggr=False)
        # self.conv5 = tg.nn.SignedConv(2, 1, first_aggr=False)

        self.lin1 = nn.Linear(13, 7)
        self.lin2 = nn.Linear(7, 4)
        self.lin3 = nn.Linear(4, 1)

    def forward(self, data):
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        x = data.x

        total_nodes = x.shape[0]

        x = x.reshape((-1, 2)).float()
        x2 = nnF.relu(self.conv1(x, edge_index, edge_weight)).float()
        x3 = nnF.relu(self.conv2(x2, edge_index, edge_weight)).float()
        x4 = nnF.relu(self.conv3(x3, edge_index, edge_weight)).float()
        x5 = nnF.relu(self.conv4(x4, edge_index, edge_weight)).float()
        x6 = nnF.relu(self.conv5(x5, edge_index, edge_weight)).float()

        res_stack = torch.cat((x, x2, x3, x4, x5, x6), 1)
        res_nn = nnF.relu(self.lin3(nnF.relu(self.lin2(nnF.relu(self.lin1(res_stack))))))

        return nnF.relu(tg.nn.global_mean_pool(res_nn.reshape(-1), data.batch))


class MeshDataset(tg.data.Dataset):
    def __init__(self, matnames):
        super(MeshDataset, self).__init__()

        self.As = []
        self.bs = []
        self.grids = []
        self.convs = []
        self.idxToMat = RangeDict()

        for i, mat in enumerate(matnames):
            cf = helpers.pickle_load_bz2(f'splittings/{mat}-cf.pkl.bz2')
            conv = helpers.pickle_load_bz2(f'splittings/{mat}-conv.pkl.bz2')
            A, b = helpers.load_recirc_flow(f'matrices/{mat}.mat')

            origI = len(self.convs)
            newI = origI + len(conv) - 1
            self.idxToMat[(origI, newI)] = i

            self.As += [A]
            self.bs += [b.flatten()]
            self.grids += cf
            self.convs += conv

    def __len__(self):
        return len(self.grids)

    def get(self, idx):
        grid = self.grids[idx]
        conv = self.convs[idx]

        gm = self.idxToMat[idx]
        A = self.As[gm]
        b = self.bs[gm]

        edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(helpers.normalize_mat(A))
        in_val = grid.astype(np.float32)
        x = torch.from_numpy(np.column_stack((grid, b))).reshape(-1, 2)
        return tg.data.Data(x=x,
                            edge_index=edge_index,
                            edge_weight=edge_weight,
                            y=torch.from_numpy(np.array([conv]).astype(np.float32)).reshape(1))


def load_model(device='cpu'):
    gnn = GNN(device).to(device)
    gnn.load_state_dict(torch.load('model/mpnn'))
    gnn.eval()
    return gnn
