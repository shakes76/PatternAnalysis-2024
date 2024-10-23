import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
import logging
from typing import Optional, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FacebookPageDataset(InMemoryDataset):
    """
    Facebook Page-Page Network Dataset using preprocessed .npz file

    The dataset contains:
    - edges: (342004, 2) connectivity matrix
    - features: (22470, 128) node feature matrix
    - target: (22470,) node labels

    Args:
        root (str): Root directory where the dataset should be saved
        val_size (float): Validation set size (default: 0.15)
        test_size (float): Test set size (default: 0.15)
        transform (callable, optional): Data transform function
        pre_transform (callable, optional): Pre-processing transform function
        force_reload (bool): Whether to reload the dataset
    """

    def __init__(
            self,
            root: str,
            val_size: float = 0.15,
            test_size: float = 0.15,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False
    ):
        self.val_size = val_size
        self.test_size = test_size
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['facebook.npz']

    @property
    def processed_file_names(self):
        return [f'facebook_processed_v{self.val_size}_{self.test_size}.pt']

    def process(self):
        """Process the .npz file into a PyTorch Geometric Data object."""
        logger.info("Processing Facebook Page-Page Network dataset...")

        # Load the .npz file
        data_path = os.path.join(self.raw_dir, 'facebook.npz')
        try:
            npz_data = np.load(data_path)

            # Convert arrays to PyTorch tensors
            x = torch.from_numpy(npz_data['features']).float()
            y = torch.from_numpy(npz_data['target']).long()

            # Convert edges to COO format for PyG
            edges = torch.from_numpy(npz_data['edges']).t().long()

            # Create train/val/test masks
            num_nodes = x.size(0)
            indices = np.arange(num_nodes)

            # First split: separate test set
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=42,
                stratify=y.numpy()
            )

            # Second split: separate train and validation sets
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_size / (1 - self.test_size),
                random_state=42,
                stratify=y[train_val_idx].numpy()
            )

            # Create boolean masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True

            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edges,
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_classes=len(torch.unique(y))
            )

            # Log dataset statistics
            logger.info(f"\nDataset statistics:")
            logger.info(f"  Number of nodes: {data.num_nodes}")
            logger.info(f"  Number of edges: {data.num_edges}")
            logger.info(f"  Feature dimensions: {data.num_features}")
            logger.info(f"  Number of classes: {data.num_classes}")
            logger.info(f"  Class distribution: {torch.bincount(data.y).tolist()}")
            logger.info(f"  Training nodes: {data.train_mask.sum().item()}")
            logger.info(f"  Validation nodes: {data.val_mask.sum().item()}")
            logger.info(f"  Test nodes: {data.test_mask.sum().item()}")

            # Save processed data
            torch.save(self.collate([data]), self.processed_paths[0])

        except Exception as e:
            logger.error(f"Error processing .npz file: {str(e)}")
            raise


def get_dataloader(
        data_dir: str,
        val_size: float = 0.15,
        test_size: float = 0.15,
        force_reload: bool = False
):
    """
    Create a dataset instance with normalized features.

    Args:
        data_dir (str): Directory containing the facebook.npz file
        val_size (float): Validation set size
        test_size (float): Test set size
        force_reload (bool): Whether to force reprocessing the data

    Returns:
        dataset: FacebookPageDataset instance
    """
    return FacebookPageDataset(
        root=data_dir,
        val_size=val_size,
        test_size=test_size,
        transform=NormalizeFeatures(),
        force_reload=force_reload
    )


if __name__ == "__main__":
    # Use the correct path
    data_dir = r"C:\Users\Ovint\Documents\PatternAnalysis-2024\recognition\GNNs4743209\data"

    try:
        # Create dataset
        logger.info("\nLoading dataset...")
        dataset = get_dataloader(data_dir, force_reload=True)
        data = dataset[0]

        # Print comprehensive dataset summary
        print("\nDataset Summary:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"Node features shape: {data.x.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Number of node features: {data.num_features}")
        print(f"Number of classes: {data.num_classes}")
        print("\nSplit sizes:")
        print(f"Training nodes: {data.train_mask.sum().item()}")
        print(f"Validation nodes: {data.val_mask.sum().item()}")
        print(f"Test nodes: {data.test_mask.sum().item()}")
        print("\nClass distribution:")
        for class_idx, count in enumerate(torch.bincount(data.y)):
            print(f"Class {class_idx}: {count.item()} nodes")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback

        traceback.print_exc()