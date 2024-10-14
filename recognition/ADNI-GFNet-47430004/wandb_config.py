'''
Weights and Biases config to avoid code in main scripts

This code was directly taken from the following github repo:
https://github.com/shakes76/GFNet
from
https://github.com/shakes76/GFNet/blob/master/wandb_config.py
'''
import wandb

def setup_wandb():
    #logging info
    #Set an experiment name to group training and evaluation
    experiment_id = wandb.util.generate_id()

    # Start a run, tracking hyperparameters
    wandb.init(
        project = "ADNI-GFNet",
        group = "GFNet",
        config={
            "id": experiment_id,
            "machine": "a100",
            "architecture": "gfnet-xs",
            "model": "GFNet",
            "dataset": "ADNI",
            "epochs": 300,
            "optimizer": "adam",
            "loss": "crossentropy",
            "metric": "accuracy",
            #~ "dim": 64,
            "depth": 12,
            "embed_dim": 384,
            "batch_size": 128
        })
    config = wandb.config