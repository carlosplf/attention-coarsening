import networkx as nx
import matplotlib.pyplot as plt

# Using the KarateClub dataset from PyTorch
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx


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
    print(f'Number of Nodes: {dataset[0].num_nodes}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    return dataset


def visualize_dataset(dataset):
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    visualize_graph(G, color=data.y)
    edge_index = data.edge_index
    print(edge_index.t())
    print(dir(data))
    edge_weight = data.edge_weight
    print(edge_weight)
