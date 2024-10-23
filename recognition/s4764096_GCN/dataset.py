import numpy as np
import dgl
import torch as th


def prepare_data():

    data = np.load('/Users/lingjieruan/Desktop/3710report/facebook_large/facebook.npz')

    edges = data['edges']
    features = data['features']
    labels = data['target']

    train_test_ratio = 0.8


    graph = dgl.graph((edges[:, 0], edges[:, 1]))

    graph.ndata['features'] = th.tensor(features, dtype=th.float32)
    graph.ndata['labels'] = th.tensor(labels, dtype=th.long)

    num_nodes = graph.num_nodes()
    train_size = int(train_test_ratio * num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[:train_size] = True
    test_mask = ~train_mask

    graph = dgl.add_self_loop(graph)

    return graph, train_mask, test_mask, features.shape[1]
