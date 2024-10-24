import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.metrics import confusion_matrix, classification_report
import networkx as nx
from torch_geometric.utils import to_networkx
import json
from typing import Dict, Tuple
import pandas as pd

from dataset import get_dataloader
from modules import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, checkpoint_path: str, data_dir: str):
        """
        Initialize predictor with model checkpoint and data.

        Args:
            checkpoint_path: Path to saved model checkpoint
            data_dir: Directory containing the dataset
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)

        # Load configuration
        config_path = self.checkpoint_path.parent / 'config.json'
        with open(config_path) as f:
            self.config = json.load(f)

        # Load dataset
        self.dataset = get_dataloader(data_dir)
        self.data = self.dataset[0].to(self.device)

        # Load model
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")

        # Create model with same configuration
        model = create_model(
            in_channels=self.dataset[0].num_features,
            num_classes=self.dataset[0].num_classes,
            model_config=self.config['model_config']
        )

        # Load state dict
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def predict(self, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions for the specified nodes.

        Args:
            mask: Boolean mask indicating which nodes to predict for
                 If None, predict for all nodes

        Returns:
            Tuple of (predictions, probabilities)
        """
        # Forward pass
        output = self.model(self.data.x, self.data.edge_index)
        probs = torch.exp(output)  # Convert log probabilities to probabilities

        if mask is not None:
            output = output[mask]
            probs = probs[mask]

        preds = output.max(1)[1]
        return preds, probs

    def analyze_performance(self, split: str = 'test'):
        """Analyze model performance on specified split."""
        logger.info(f"\nAnalyzing performance on {split} set:")

        # Get appropriate mask
        if split == 'test':
            mask = self.data.test_mask
        elif split == 'val':
            mask = self.data.val_mask
        else:
            mask = self.data.train_mask

        # Get predictions
        preds, probs = self.predict(mask)
        true_labels = self.data.y[mask]

        # Create save directory for plots
        save_dir = self.checkpoint_path.parent / 'analysis'
        save_dir.mkdir(exist_ok=True)

        # 1. Classification Report
        logger.info("\nClassification Report:")
        report = classification_report(
            true_labels.cpu(),
            preds.cpu(),
            output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        logger.info(f"\n{report_df}")

        # 2. Confusion Matrix
        self.plot_confusion_matrix(true_labels, preds, save_dir / f'confusion_matrix_{split}.png')

        # 3. Prediction Confidence Distribution
        self.plot_confidence_distribution(probs, preds, true_labels,
                                          save_dir / f'confidence_dist_{split}.png')

        # 4. Node Embedding Visualization
        self.plot_node_embeddings(mask, save_dir / f'embeddings_{split}.png')

        # 5. Misclassification Analysis
        self.analyze_misclassifications(true_labels, preds, probs, mask)

        return report_df

    def plot_confusion_matrix(self, true_labels, preds, save_path):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_labels.cpu(), preds.cpu())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path)
        plt.close()

    def plot_confidence_distribution(self, probs, preds, true_labels, save_path):
        """Plot distribution of prediction confidences."""
        confidences = probs.max(1)[0].cpu()
        correct = (preds == true_labels).cpu()

        plt.figure(figsize=(10, 6))
        plt.hist([confidences[correct], confidences[~correct]],
                 label=['Correct', 'Incorrect'], bins=20, alpha=0.6)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_node_embeddings(self, mask, save_path):
        """Plot node embeddings using t-SNE."""
        from sklearn.manifold import TSNE

        # Get node embeddings (before final classification layer)
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeddings(self.data.x, self.data.edge_index)[mask]

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.cpu())

        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=self.data.y[mask].cpu(), cmap='tab20')
        plt.colorbar(scatter)
        plt.title('Node Embeddings (t-SNE)')
        plt.savefig(save_path)
        plt.close()

    def analyze_misclassifications(self, true_labels, preds, probs, mask):
        """Analyze characteristics of misclassified nodes."""
        incorrect = preds != true_labels

        if incorrect.sum() == 0:
            logger.info("No misclassifications found!")
            return

        # Get node degrees
        graph = to_networkx(self.data)
        degrees = torch.tensor([deg for _, deg in graph.degree()])
        degrees = degrees[mask]

        # Analyze node degrees for correct vs incorrect predictions
        avg_degree_correct = degrees[~incorrect].float().mean()
        avg_degree_incorrect = degrees[incorrect].float().mean()

        logger.info("\nMisclassification Analysis:")
        logger.info(f"Number of misclassified nodes: {incorrect.sum().item()}")
        logger.info(f"Average node degree (correct): {avg_degree_correct:.2f}")
        logger.info(f"Average node degree (incorrect): {avg_degree_incorrect:.2f}")

        # Analyze confidence of misclassifications
        conf_incorrect = probs.max(1)[0][incorrect]
        logger.info(f"Average confidence of misclassifications: {conf_incorrect.mean():.3f}")

    def predict_new_nodes(self, new_features: torch.Tensor, new_edges: torch.Tensor):
        """
        Make predictions for new nodes.

        Args:
            new_features: Feature matrix for new nodes
            new_edges: Edge index matrix connecting new nodes to existing graph

        Returns:
            Predictions and probabilities for new nodes
        """
        self.model.eval()
        with torch.no_grad():
            # Combine new features with existing graph
            combined_features = torch.cat([self.data.x, new_features])
            combined_edges = torch.cat([self.data.edge_index, new_edges], dim=1)

            # Forward pass
            output = self.model(combined_features, combined_edges)

            # Get predictions for new nodes
            new_output = output[-len(new_features):]
            probs = torch.exp(new_output)
            preds = new_output.max(1)[1]

        return preds, probs


def main():
    # Paths
    checkpoint_path = r".\runs\run_20241024_103409\best_model.pth"
    data_dir = r".\data"

    try:
        # Initialize predictor
        predictor = ModelPredictor(checkpoint_path, data_dir)

        # Analyze performance on all splits
        for split in ['train', 'val', 'test']:
            predictor.analyze_performance(split)
            predictor.plot_node_embeddings(predictor.data.test_mask, Path('output') / f'embeddings_{split}.png')
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise


if __name__ == "__main__":
    main()