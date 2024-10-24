import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import get_dataloader
from modules import create_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
            self,
            model,
            data,
            lr=0.001,
            weight_decay=5e-4,
            patience=10,
            max_epochs=200,
            save_dir='checkpoints'
    ):
        """
        Initialize the trainer.

        Args:
            model: The GNN model
            data: PyG data object
            lr: Learning rate
            weight_decay: Weight decay for regularization
            patience: Patience for early stopping
            max_epochs: Maximum number of epochs
            save_dir: Directory to save checkpoints
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)
        self.patience = patience
        self.max_epochs = max_epochs

        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking variables
        self.best_val_f1 = 0
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        # Initialize history
        self.history = {
            'train_loss': [],
            'train_f1': [],
            'val_loss': [],
            'val_f1': [],
            'test_f1': [],
            'lr': []
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(self.data.x, self.data.edge_index)

        # Compute loss only on training nodes
        loss = F.nll_loss(output[self.data.train_mask], self.data.y[self.data.train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        train_f1 = self.compute_f1(output[self.data.train_mask], self.data.y[self.data.train_mask])

        return loss.item(), train_f1

    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate the model on the given mask."""
        self.model.eval()

        # Forward pass
        output = self.model(self.data.x, self.data.edge_index)

        # Compute loss and metrics
        loss = F.nll_loss(output[mask], self.data.y[mask])
        f1 = self.compute_f1(output[mask], self.data.y[mask])

        return loss.item(), f1

    def compute_f1(self, output, targets):
        """Compute F1 score."""
        preds = output.max(1)[1].cpu().numpy()
        targets = targets.cpu().numpy()
        return f1_score(targets, preds, average='weighted')

    def save_checkpoint(self, epoch, filename='best_model.pth'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def plot_training_history(self):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('Training and Validation Loss')

        # Plot F1 scores
        ax2.plot(self.history['train_f1'], label='Train F1')
        ax2.plot(self.history['val_f1'], label='Validation F1')
        ax2.plot(self.history['test_f1'], label='Test F1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.set_title('F1 Scores')

        # Save plot
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")

        for epoch in range(self.max_epochs):
            # Training
            train_loss, train_f1 = self.train_epoch()

            # Validation
            val_loss, val_f1 = self.evaluate(self.data.val_mask)

            # Test
            test_loss, test_f1 = self.evaluate(self.data.test_mask)

            # Update learning rate
            self.scheduler.step(val_f1)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)
            self.history['test_f1'].append(test_f1)
            self.history['lr'].append(current_lr)

            # Log progress
            logger.info(f"Epoch {epoch:03d}:")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            logger.info(f"  Test F1: {test_f1:.4f}")
            logger.info(f"  Learning Rate: {current_lr}")

            # Check for improvement
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement == self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        # Final evaluation
        self.plot_training_history()
        logger.info("\nTraining completed!")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f} at epoch {self.best_epoch}")

        return self.history


def main():
    # Configuration
    config = {
        'data_dir': r".\data",
        'model_config': {
            'hidden_channels': 256,
            'num_layers': 8,
            'dropout': 0.5,
            'conv_type': 'GCN',
            'heads': 1
        },
        'train_config': {
            'lr': 0.005,
            'weight_decay': 5e-4,
            'patience': 30,
            'max_epochs': 300
        }
    }

    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"runs/run_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info("Loading dataset...")
        dataset = get_dataloader(config['data_dir'])
        data = dataset[0]

        # Create model
        logger.info("Creating model...")
        model = create_model(
            in_channels=dataset[0].num_features,
            num_classes=dataset[0].num_classes,
            model_config=config['model_config']
        )

        # Create trainer and train
        trainer = Trainer(
            model=model,
            data=data,
            save_dir=save_dir,
            **config['train_config']
        )

        history = trainer.train()

        # Save configuration
        import json
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=4)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
