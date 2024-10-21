from utils import *
from settings import *
from dataset import get_dataloader
from modules import Discriminator, Generator


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    """
    Calculates gradient penalty for WGAN-GP loss.
    """
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic (discriminator) scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def train_fn(
    critic: Discriminator,
    gen: Generator,
    loader: DataLoader,
    dataset: Dataset,
    step,
    alpha,
    opt_critic: optim.Adam,
    opt_gen: optim.Adam,
):
    """
    Main training function.
    """
    loop = tqdm(loader, leave=True)

    # Keep track of loss for visualisation
    gen_loss = []
    critic_loss = []

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

        fake = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, device=DEVICE)
        loss_critic = (
            - (torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001 * torch.mean(critic_real ** 2))
        )

        # Backpropagate and optimize critic
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        # Backpropagate and optimize generator
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / ((PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        loop.set_postfix(gp=gp.item(), loss_critic=loss_critic.item())

        # Extend loss lists
        gen_loss.append(loss_gen.item())
        critic_loss.append(loss_critic.item())

    return alpha, gen_loss, critic_loss


def plot_losses(gen_loss, critic_loss, step):
    """
    TODO: fix
    """
    plt.figure(figsize=(10, 5))
    plt.plot(gen_loss, label="Generator Loss", color="orange")
    plt.plot(critic_loss, label="Critic Loss", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists(f"{SRC}/loss_plots/{MODEL_LABEL}"):
        os.makedirs(f"{SRC}/loss_plots/{MODEL_LABEL}")
    plt.savefig(f"{SRC}/loss_plots/{MODEL_LABEL}/loss_plot_step{step}.png")


def train():
    """
    Trains and saves a StyleGAN model on a given dataset.
    """

    # Initialize main models and optimizers
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    critic = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
    opt_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                        {"params": gen.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

    # Set models to training mode
    gen.train()
    critic.train()

    # Initialize storage for losses
    cum_gen_loss = []
    cum_critic_loss = []

    # start at step that corresponds to img size that we set in config
    step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5   # start with very low alpha
        loader, dataset = get_dataloader(4 * 2 ** step)  
        print(f"Current image size: {4 * 2 ** step}")

        step_gen_loss = []
        step_critic_loss = []

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            alpha, gen_loss, critic_loss = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen
            )

            # Extend cumulative loss lists
            step_gen_loss.extend(gen_loss)
            step_critic_loss.extend(critic_loss)
            cum_gen_loss.extend(gen_loss)
            cum_critic_loss.extend(critic_loss)

        generate_examples(gen, step)
        plot_losses(step_gen_loss, step_critic_loss, step)
        step += 1  # progress to the next img size

    # Save models
    if not os.path.exists(f"{SRC}/saved_models"):
        os.makedirs(f"{SRC}/saved_models")
    torch.save(gen.state_dict(), f"{SRC}/saved_models/gen_{MODEL_LABEL}.pt")
    torch.save(critic.state_dict(), f"{SRC}/saved_models/critic_{MODEL_LABEL}.pt")


if __name__ == "__main__":
    t1 = time.time()  # start time
    print("Beginning training...")
    train()
    t2 = time.time()  # end time
    print(f"Training complete. Time taken: {t2-t1:.2f} seconds.")
    print("The models have been saved and can be found at '<SRC>/saved_models/<MODEL_LABEL>'")
