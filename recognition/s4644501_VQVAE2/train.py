"""
Training script for VQVAE model on HIPMRI prostate
cancer dataset.

@author George Reid-Smith
"""
import logging
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from dataset import MRIDataloader
from modules import VQVAE
from metrics import avg_ssim, avg_loss, batch_ssim
from utils import load_config, setup_logging, save_samples, plot_results, save_model

class VQVAETrainer:
    def __init__(self, config):
        self._config(config)

        self.device = self._get_device()
        self.model = VQVAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.max_ssim = 0
        self.iterations = []
        self.training_losses = []
        self.training_ssims = []
        self.validation_losses = []
        self.validation_ssims = []

        self._setup_data_loaders()
    
    def _config(self, config):
        """Configures class attributes for the model from 
        a configuration dictionary.

        Args:
            config (dict): dictionary loaded from configuration
            yaml.
        """
        # Data directories
        self.train_path = config['input']['train_dir']
        self.validate_path = config['input']['validate_dir']
        self.test_path = config['input']['test_dir']

        # Model parameters
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.lr = config['training']['lr']
        self.llw = config['training']['latent_loss_weight']
        self.sample_size = config['training']['sample_size']
        self.interval = config['training']['interval']

        # Output directories
        self.log_dir = config['output']['log_dir']
        self.sample_dir = config['output']['sample_dir']
        self.visual_dir = config['output']['visual_dir']
        self.model_dir = config['output']['model_dir']
        
    def _get_device(self):
        """Load device based on GPU cuda availability.

        Returns:
            torch.device: the training device
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Using CPU')
        
        return device

    def _setup_data_loaders(self):
        """Build the train and validation dataloaders.
        """
        self.train_loader = MRIDataloader(self.train_path, self.batch_size, shuffle=True).load()
        self.validate_loader = MRIDataloader(self.validate_path, self.batch_size, shuffle=False).load()
  
        logging.info('Dataloaders Initialized')
    
    def _save_results(self, images, outputs, training_loss, training_ssim, total_iterations, epoch):
        """Helper function to log and output training results at 
        specified intervals.

        Args:
            images (torch.Tensor): batch image tensors
            outputs (torch.Tensor): reconstructed image tensors
            training_loss (float): training reconstruction loss
            training_ssim (float): training ssim
            total_iterations (int): total iterations elapsed in training
            epoch (int): current training epoch
        """
        self.iterations.append(total_iterations)

        # Average training loss and ssim
        avg_training_loss = sum(training_loss) / len(training_loss)
        avg_training_ssim = sum(training_ssim) / len(training_ssim)
        self.training_losses.append(avg_training_loss)
        self.training_ssims.append(avg_training_ssim)

        # Average validation loss and ssim
        avg_validation_loss = avg_loss(self.device, self.model, self.criterion, self.llw, self.validate_loader)
        avg_validation_ssim = avg_ssim(self.device, self.model, self.validate_loader)
        self.validation_losses.append(avg_validation_loss)
        self.validation_ssims.append(avg_validation_ssim)

        logging.info(f'Epoch: {epoch}/{self.epochs} Training Loss: {avg_training_loss:.4f} Validation Loss: {avg_validation_loss:.4f} Training SSIM: {avg_training_ssim:.4f} Validation SSIM: {avg_validation_ssim:.4f}')

        # Save model if best performance
        if avg_validation_ssim > self.max_ssim:
            save_model(epoch, avg_validation_ssim, self.model, self.optimizer, self.model_dir)
            
            logging.info(f'Saved Model')

        # Save training samples
        save_samples(images, outputs, self.sample_size, self.sample_dir, title=f'epoch_{epoch}_iter_{total_iterations}')

        logging.info('Saved Training Samples')

    def _training_step(self, images):
        """Forward pass and back-propogation for a single training batch.

        Args:
            images (torch.Tensor): input image tensor

        Returns:
            _type_: _description_
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs, latent_loss = self.model(images)

        # Calculate batch loss
        recon_loss = self.criterion(outputs, images)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.llw * latent_loss
    
        # Calculate batch SSIM
        ssim = batch_ssim(images, outputs, self.device)

        # Back propagate
        loss.backward()
        self.optimizer.step()

        return outputs, loss, ssim

    def train(self):
        """Training loop for the VQVAE.
        """
        logging.info('Training VQVAE')

        # Training metrics
        total_iterations = 0
        
        for epoch in range(self.epochs):
            self.model.train()

            interval_training_losses = []
            interval_training_ssims = []

            for i, images in enumerate(self.train_loader):
                total_iterations += 1
                images = images.to(self.device)

                outputs, loss, ssim = self._training_step(images)
                interval_training_losses.append(loss.item())
                interval_training_ssims.append(ssim.item())

                if i % self.interval  == 0:
                    self._save_results(images, outputs, interval_training_losses, interval_training_ssims, total_iterations, epoch+1)
        
        plot_results(self.iterations, self.training_losses, self.training_ssims, self.validation_losses, self.validation_ssims, self.visual_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()

    # Load configuration yaml into a dictionary
    config = load_config(args.config)
    setup_logging(config['output']['log_dir'], 'training.log')

    trainer = VQVAETrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
