import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNBlock(nn.Module):
    """
    A configurable GNN block that supports different types of graph convolutions.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            conv_type: str = 'GCN',
            dropout: float = 0.5,
            heads: int = 1,
            residual: bool = True
    ):
        super().__init__()
        self.conv_type = conv_type
        self.residual = residual
        self.same_dims = in_channels == out_channels

        # Choose convolution type
        if conv_type == 'GAT':
            self.conv = GATConv(
                in_channels,
                out_channels // heads,
                heads=heads,
                dropout=dropout
            )
        elif conv_type == 'SAGE':
            self.conv = SAGEConv(in_channels, out_channels)
        else:  # Default to GCN
            self.conv = GCNConv(in_channels, out_channels)

        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions differ
        if residual and not self.same_dims:
            self.res_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        identity = x

        x = self.conv(x, edge_index)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.residual:
            if self.same_dims:
                x = x + identity
            else:
                x = x + self.res_proj(identity)

        return x


class FacebookGNN(nn.Module):
    """
    GNN model for Facebook page classification.
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            num_classes: int,
            num_layers: int = 3,
            dropout: float = 0.5,
            conv_type: str = 'GCN',
            heads: int = 4
    ):
        super().__init__()
        self.num_layers = num_layers

        # Input layer
        self.input_block = GNNBlock(
            in_channels,
            hidden_channels,
            conv_type=conv_type,
            dropout=dropout,
            heads=heads
        )

        # Hidden layers
        self.layers = nn.ModuleList([
            GNNBlock(
                hidden_channels,
                hidden_channels,
                conv_type=conv_type,
                dropout=dropout,
                heads=heads,
                residual=True
            ) for _ in range(num_layers - 2)
        ])

        # Final MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes)
        )

        # Initialize MLP weights
        self._init_mlp_weights()

    def _init_mlp_weights(self):
        """Initialize MLP weights using Xavier initialization."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings before the final classification layer.

        Args:
            x (Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges]

        Returns:
            Tensor: Node embeddings with shape [num_nodes, hidden_channels]
        """
        # Input block
        x = self.input_block(x, edge_index)

        # Hidden layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Apply first part of MLP to get final embeddings
        embeddings = self.mlp[0](x)  # Linear layer
        embeddings = self.mlp[1](embeddings)  # ReLU activation

        return embeddings

    def forward(self, x, edge_index):
        """Forward pass of the model."""
        # Input block
        x = self.input_block(x, edge_index)

        # Hidden layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Final prediction
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)


def create_model(
        in_channels: int,
        num_classes: int,
        model_config: dict = None
) -> FacebookGNN:
    """Create a GNN model with the specified configuration."""
    default_config = {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.5,
        'conv_type': 'GAT',
        'heads': 4
    }

    # Update default config with provided config
    if model_config:
        default_config.update(model_config)

    logger.info("Creating model with configuration:")
    for key, value in default_config.items():
        logger.info(f"  {key}: {value}")

    return FacebookGNN(
        in_channels=in_channels,
        num_classes=num_classes,
        **default_config
    )


if __name__ == "__main__":
    # Example usage
    in_channels = 128  # Feature dimension from your dataset
    num_classes = 4  # Number of classes in your dataset

    # Test with smaller network for demonstration
    x = torch.randn(100, in_channels)  # 100 nodes
    edge_index = torch.randint(0, 100, (2, 300))  # 300 edges

    # Test each model type
    for conv_type in ['GCN', 'GAT', 'SAGE']:
        logger.info(f"\nTesting {conv_type} model:")
        try:
            model = create_model(
                in_channels=in_channels,
                num_classes=num_classes,
                model_config={'conv_type': conv_type}
            )

            # Test forward pass
            out = model(x, edge_index)
            logger.info(f"Output shape: {out.shape}")
            logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

        except Exception as e:
            logger.error(f"Error testing {conv_type} model: {str(e)}")
            import traceback

            traceback.print_exc()