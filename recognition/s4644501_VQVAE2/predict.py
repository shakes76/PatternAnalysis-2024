"""
Runs prediction on a VQVAE model.

@author George Reid-Smith
"""
import argparse
import logging

import torch

from modules import VQVAE
from utils import load_config, save_samples, setup_logging
from metrics import avg_ssim
from dataset import MRIDataloader

class VQVAEPredict:
    """Runs inference on a trained VQVAE.
    """
    def __init__(self, config):
        self._config(config)
        self.device = self._get_device()
        self.model = VQVAE().to(self.device)

        self._load_model()
        self._setup_test_loader()
    
    def _config(self, config):
        """Configures class attributes for the model from 
        a configuration dictionary.

        Args:
            config (dict): dictionary loaded from configuration
            yaml.
        """
        self.test_dir = config['input']['test_dir']
        self.model_dir = config['input']['load_model_dir']
        self.samples = config['predict']['samples']
        self.sample_size = config['predict']['sample_size']
        self.batch_size = config['predict']['batch_size']
        self.log_dir = config['output']['log_dir']
        self.sample_dir = config['output']['sample_dir']
        self.visual_dir = config['output']['visual_dir']

    def _get_device(self):
        """Load device based on GPU cuda availability.

        Returns:
            torch.device: the training device
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'Using CPU')
        
        return device
    
    def _load_model(self):
        """Loads VQVAE model from directory in config.
        """
        checkpoint = torch.load(self.model_dir, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logging.info('Model Loaded')

    def _setup_test_loader(self):
        """Build the test dataloader.
        """
        self.test_loader = MRIDataloader(self.test_dir, self.batch_size, shuffle=True).load()
        
        logging.info('Dataloader Initiated')

    def predict(self):
        """Run inference on the loaded model. Log the SSIM on the test batch.
        """
        test_ssim = avg_ssim(self.device, self.model, self.test_loader)        
        logging.info(f'SSIM on Test Dataset: {test_ssim:.4f}')
        
        samples = iter(self.test_loader)

        for i in range(3):
            images = next(samples).to(self.device)
            
            outputs, _ = self.model(images)

            save_samples(images, outputs, self.sample_size, self.sample_dir, f'predict_set_{i}')
                    
        logging.info('Saved Prediction Samples')
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to predict config file')
    args = parser.parse_args()
    
    # Load configuration yaml into a dictionary
    config = load_config(args.config)
    setup_logging(config['output']['log_dir'], 'predict.log')

    predict_model = VQVAEPredict(config)
    predict_model.predict()

if __name__ == '__main__':
    main()
