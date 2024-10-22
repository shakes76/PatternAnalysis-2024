"""
Weights and Biases config file
Structure based off Shakes' config file
"""
import wandb
def setup_wandb(architecture: str, epochs: int, decay, lr):
    """
    Initialises a wandb run with respective model information.
    """
    # Log run with name that desceibes appropraite hyperparameters
    experiment_id = wandb.util.generate_id()
    name = f"{architecture}, {epochs} epochs, {decay} decay, {lr} learning rate"

    # Start a run, tracking hyperparameters
    wandb.init(
        project="Facebook-Large-Page-Page-Network",
        group="GNN",
        name=name,
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
