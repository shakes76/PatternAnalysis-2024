import torch
import matplotlib.pyplot as plt
from einops import rearrange
from timm.utils import ModelEmaV3
import modules


def display_reverse(images):
    """
    Display reverse process of image generation
    :param images: final generated samples
    :return:
    """
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()


def generate_images(checkpoint_path = None, num_time_steps = 1000, ema_decay = 0.9999, ):
    """
    Generate sample images
    :param checkpoint_path: model checkpoint directory
    :param num_time_steps: number of steps set at 1000
    :param ema_decay: exponential moving average decay
    """
    checkpoint = torch.load(checkpoint_path)
    model = modules.UNET().cuda()
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = modules.DiffusionScheduler(num_time_steps=num_time_steps)
    # time snapshots for plotting
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 3, 128, 128)
            for t in reversed(range(1, num_time_steps)):
                t = [t]
                temp = (scheduler.beta[t] / (
                            (torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
                z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (temp * model(z.cuda(), t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 3, 128, 128)
                z = z + (e * torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
            x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * z - (temp * model(z.cuda(), [0]).cpu())

            # reverse normalisation to construct original-like images
            mean = torch.tensor([0.1798, 0.1798, 0.1798]).view(1, 3, 1, 1)
            std = torch.tensor([0.1964, 0.1964, 0.1964]).view(1, 3, 1, 1)
            x = (x * std) + mean

            images.append(x)

            # Convert to numpy and visualize
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach().numpy()
            plt.imshow(x)
            plt.show()

            display_reverse(images)
            images = []


if __name__ == "__main__":
    generate_images('ddpm_checkpoint')
