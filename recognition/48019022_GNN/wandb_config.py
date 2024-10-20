"""
Weights and Biases config file
Structure based off Shakes' config file

To be completed.
"""
import wandb

def setup_wandb():
    #logging info
    #Set an experiment name to group training and evaluation
    experiment_id = wandb.util.generate_id()

    # Start a run, tracking hyperparameters
    wandb.init(
        project="Facebook-Large-Page-Page-Network",
        group="GNN",
        # track hyperparameters and run metadata
        config={
        #     "id": experiment_id,
        #     "machine": "A100",
        #     "architecture": "GCN",
        #     "model": "GNN",
        #     "dataset": "ImageNet",
        #     "epochs": 300,
        #     "optimizer": "adam",
        #     "loss": "crossentropy",
        #     "metric": "accuracy",
        #     #~ "dim": 64,
        #     "depth": 12,
        #     "embed_dim": 384,
        #     "batch_size": 128
        })
    # config = wandb.config
