"""
Weights and Biases config file
Structure based off Shakes' config file

To be completed.
"""
import wandb

import wandb

def setup_wandb(architecture: str, epochs: int):
    """
    Initialises a wandb run with respective model information.
    """
    experiment_id = wandb.util.generate_id()

    # Start a run, tracking hyperparameters
    wandb.init(
        project="Facebook-Large-Page-Page-Network",
        group="GNN",
        config={
            "id": experiment_id,
            "architecture": architecture,
            "model": "GNN",
            "dataset": "Facebook",
            "epochs": epochs,
            "optimizer": "adam",
            "loss": "crossentropy",
            "metric": "accuracy",
        })
    config = wandb.config
