import time
import numpy as np
import torch
import torch.optim as optim

from gcn_model.gcn import GCN
from gat_model.gat import GAT

import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from utils import graph_tools


EPOCHS = 20


def train_model(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def train(model, optimizer, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # Transposing data.y
    data_y_transp = data.y.permute(*torch.arange(data.y.ndim - 1, -1, -1))

    features, labels = Variable(data.x), Variable(data_y_transp)
    adj = build_adj(data)

    output = model(features, adj)

    idx_train = range(20)
    idx_val = range(20, 28)
    idx_test = range(28, 34)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def build_adj(data):
    edges = data.edge_index

    # data.y transpose, replacing data.y.T
    labels = data.y.permute(*torch.arange(data.y.ndim - 1, -1, -1))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    adj = torch.FloatTensor(np.array(adj.todense()))

    return adj


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def build_model():
    # nheads define the number of attention layers
    model = GAT(nfeat=34,
                nhid=1,
                nclass=4,
                dropout=0.6,
                nheads=1,
                alpha=0.2)
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=5e-4
    )
    return model, optimizer


if __name__ == "__main__":
    data = graph_tools.load_dataset()
    adj = build_adj(data)

    model, optimizer = build_model()

    t_total = time.time()
    loss_values = []

    for epoch in range(EPOCHS):
        loss_values.append(train(model, optimizer, epoch))

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    a_layer = model.get_att_layers()
    e_matrix = a_layer[0].get_e_instance()
    print(f'Size of e matrix: {len(e_matrix)} x {len(e_matrix[0])}')
    print(e_matrix)