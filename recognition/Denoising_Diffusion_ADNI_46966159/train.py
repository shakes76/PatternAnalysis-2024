import torch
import torch.nn as nn
import torch.optim as optim
import modules
import matplotlib.pyplot as plt
from timm.utils import ModelEmaV3
from tqdm import tqdm
import dataset as ds

def train(batch_size = ds.batch_size, num_time_steps = 1000, num_epochs = 15, ema_decay = 0.9999, epsilon = 2e-5,
          checkpoint_path: str=None):
    """
    Entire training pipeline using 1000 steps, 15 epochs, learning rate 2e-5 and ema of 0.9999
    :param batch_size: Training batch size
    :param num_time_steps: number tiem steps in diffusion process
    :param num_epochs: number of epochs to train for
    :param ema_decay: exponential moving average decay
    :param epsilon: learning rate
    :param checkpoint_path: directory to save model to
    """

    # save losses for plot
    losses = []
    scheduler = modules.DiffusionScheduler(num_time_steps=num_time_steps)
    # use cuda gpu always
    model = modules.UNET().cuda()
    optimizer = optim.Adam(model.parameters(), lr=epsilon)
    ema = ModelEmaV3(model, decay=ema_decay)
    # load previously saved trained model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    # mean squares error loss
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(ds.dataloader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.cuda()
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
            output = model(x, t)
            optimizer.zero_grad()
            loss = criterion(output, e)
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('loss.png')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    torch.save(checkpoint, '/content/ddpm_checkpoint')

train(checkpoint_path=None, epsilon=2e-5, num_epochs=100)