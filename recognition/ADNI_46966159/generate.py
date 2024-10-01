import torch
import dataset, models

# Loading the trained model
best_model = models.Diffusion(models.UNet(), n_steps=10000, device=dataset.device)
best_model.load_state_dict(torch.load("diffusion.pt", map_location=dataset.device))
best_model.eval()
print("Model loaded")

print("Generating new images")
generated = models.generate_new_images(
    best_model,
    n_samples=100,
    device=dataset.device
)
dataset.show_images(generated, "Final result")