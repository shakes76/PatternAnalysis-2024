import logging
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms, utils
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from dataset import MRIDataset
from modules import VQVAE
from metrics import calculate_ssim
from helpers import load_config, setup_logging, save_samples, plot_ssim

class VQVAETrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.model = VQVAE().to(self.device)
        self.batch_size = config['training']['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.criterion = nn.MSELoss()
        self.latent_loss_weight = config['training']['latent_loss_weight']
        self.sample_size = config['training']['sample_size']
        self.epochs = config['training']['epochs']

        self._setup_data_loaders()
    
    def _get_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Using CPU')
        
        return device

    def _setup_data_loaders(self):
        train_dataset = MRIDataset(self.config['data']['train_path'])
        validate_dataset = MRIDataset(self.config['data']['validate_path'])
        test_dataset = MRIDataset(self.config['data']['test_path'])

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
    
    def train(self):
        logging.info('Training')

        total_iterations = 0
        ssim_scores = []
        iterations = []

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
                loss = recon_loss + self.latent_loss_weight * latent_loss

                # Back propogation
                loss.backward()
                self.optimizer.step()

                if i % self.config['training']['interval'] == 0:
                    ssim = self.validate(epoch)
                    ssim_scores.append(ssim)
                    iterations.append(total_iterations)

                    save_samples(epoch, i, self.model, images, self.sample_size, self.device, self.config['logging']['sample_dir'])
                    self.model.train()

        plot_ssim(iterations, ssim_scores, self.config['logging']['visual_dir'])

    def validate(self, epoch):
        self.model.eval()
        avg_ssim = calculate_ssim(self.device, self.model, self.validate_loader)
        logging.info(f'Epoch {epoch+1}/{self.epochs}\t Average SSIM on validation: {avg_ssim:.4f}')

        return avg_ssim

    def test(self):
        self.model.eval()
        avg_ssim = calculate_ssim(self.device, self.model, self.test_loader)
        logging.info(f'Average SSIM on test: {avg_ssim:.4f}')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/local.yaml', help='Path to config file.')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config['logging']['log_dir'])

    trainer = VQVAETrainer(config)
    trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()