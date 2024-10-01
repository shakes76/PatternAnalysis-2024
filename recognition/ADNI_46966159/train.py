import torch
import torch.nn as nn
import dataset as ds
import models as m

def training_loop(diffusion, dataloader, n_epochs, optim, device, display=False, dir="diffusion.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = diffusion.n_steps

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            print(f"Epoch {epoch + 1}/{n_epochs}")
            # Loading data
            x_0 = batch[0].to(device)
            n = len(x_0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            zeta = torch.randn_like(x_0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Computing the noisy image based on x_0 and the time-step (forward process)
            noisy_imgs = diffusion(x_0, t, zeta)

            # Getting model estimation of noise based on the images and the time-step
            zeta_tau = diffusion.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(zeta_tau, zeta)
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x_0) / len(dataloader.dataset)

        # Display images generated at this epoch
        # if display:
        #     dataset.show_images(models.generate_new_images(diffusion, device=device), f"Images generated at epoch {epoch + 1}")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(diffusion.state_dict(), dir)

        print(log_string)

# Defining model
n_steps, min_beta, max_beta = 10000, 10 ** -4, 0.02  # More steps than paper
ddpm = m.Diffusion(m.UNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=ds.device)

m.show_forward(ddpm, ds.dataloader, ds.device)


optimizer = torch.optim.Adam(ddpm.parameters(), lr=0.001)
training_loop(ddpm, ds.dataloader, ds.n_epochs, optimizer, ds.device)

generated = m.generate_new_images(ddpm)
ds.show_images(generated, "Images")

