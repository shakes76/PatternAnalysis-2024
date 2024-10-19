import logging
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from dataset import MRIDataset
from modules import VQVAE
from metrics import calculate_ssim
from utils import load_config, setup_logging, save_samples, plot_results, save_model

#TODO include loss plot as well as SSIM.

class VQVAETrainer:
    def __init__(self, config):
        self._config(config)
        self.device = self._get_device()
        self.model = VQVAE().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self._setup_data_loaders()
    
    def _config(self, config):
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Using CPU')
        
        return device

    def _setup_data_loaders(self):
        train_dataset = MRIDataset(self.train_path)
        validate_dataset = MRIDataset(self.validate_path)
        test_dataset = MRIDataset(self.test_path)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        self.validate_loader = DataLoader(
            validate_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        logging.info('Dataloaders initialized')
        
    def train(self):
        logging.info('Training')

        total_iterations = 0
        ssim_scores = []
        losses = []
        iterations = []
        
        max_ssim = 0

        for epoch in range(self.epochs):
            self.model.train()

            for i, images in enumerate(self.train_loader):
                total_iterations += 1

                images = images.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, latent_loss = self.model(images)

                # Calculate loss
                recon_loss = self.criterion(outputs, images)
                latent_loss = latent_loss.mean()
                loss = recon_loss + self.llw * latent_loss

                # Back propogation
                loss.backward()
                self.optimizer.step()

                if i % self.interval  == 0:
                    ssim = self.validate(epoch, loss)
                    ssim_scores.append(ssim)
                    losses.append(loss.cpu())
                    iterations.append(total_iterations)

                    # Save best model
                    if ssim > max_ssim:
                        max_ssim = ssim
                        save_model(epoch, ssim, self.model, self.optimizer, self.model_dir)
                    
                    # Save forward passed samples from current model
                    save_samples(
                        images=images,
                        sample_size=self.sample_size,
                        device=self.device,
                        model=self.model,
                        output_dir=self.sample_dir,
                        title=f'epoch_{epoch}_iter_{i}'
                    )

        plot_results(iterations, ssim_scores, losses, self.visual_dir)

    def validate(self, epoch, loss):
        self.model.eval()
        avg_ssim = calculate_ssim(self.device, self.model, self.validate_loader)
        logging.info(f'Epoch {epoch+1}/{self.epochs}\t Average SSIM on validation: {avg_ssim:.4f}\t Current Model Loss: {loss}')

        return avg_ssim

    def test(self):
        self.model.eval()
        avg_ssim = calculate_ssim(self.device, self.model, self.test_loader)
        logging.info(f'Average SSIM on test: {avg_ssim:.4f}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config['output']['log_dir'])

    trainer = VQVAETrainer(config)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()
