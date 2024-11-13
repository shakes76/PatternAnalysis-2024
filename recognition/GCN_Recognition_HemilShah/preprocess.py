from torch_geometric.utils import add_remaining_self_loops

def preprocess(graph_data):
    # Ensure the edge index is in the correct shape [2, E]
    if graph_data.edge_index.shape[0] != 2:
        graph_data.edge_index = graph_data.edge_index.T

    num_nodes = graph_data.x.size(0)

    # Add self-loops safely
    edge_index, _ = add_remaining_self_loops(
        graph_data.edge_index, num_nodes=num_nodes
    )

    graph_data.edge_index = edge_index  # Update the edge index
    return graph_data
