import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx
from gcn_model.gcn import GCN


# check PyTorch installation
os.environ['TORCH'] = torch.__version__
print(torch.__version__)


def visualize_graph(G, color):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


def load_dataset():
    dataset = KarateClub()
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    return dataset


def visualize_dataset(dataset):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)


def train_model(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    optimizer.zero_grad()
    out, h = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, h


if __name__ == "__main__":
    dataset = load_dataset()
    # visualize_dataset(dataset=dataset)

    data = dataset[0]
    model = GCN(dataset)

    for epoch in range(801):
        loss, h = train_model(model, data)
        if epoch % 10 == 0:
            print(loss.item())

    visualize_embedding(h, color=data.y, epoch=epoch, loss=loss)
