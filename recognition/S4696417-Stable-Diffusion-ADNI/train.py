from torchvision import transforms
from dataset import get_dataloader
from modules import StableDiffusion
import torch, wandb, os
import torch.nn as nn
from tqdm import tqdm

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cwd = os.getcwd()

train_loader_AD = get_dataloader(os.path.join(cwd,'data/train/AD'), batch_size=16, transform=image_transform)
train_loader_CN = get_dataloader(os.path.join(cwd,'data/train/NC'), batch_size=16, transform=image_transform)

# Settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

lr = 1e-5
epochs = 100

model = StableDiffusion(
    in_channels=3,
    out_channels=3,
    model_channels=256,
    num_res_blocks=2,
    attention_resolutions=[16,8],
    channel_mult=[1, 2, 4, 8],
    num_heads=8
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# initialise wandb
wandb.init(
    project="S4696417-Stable-Diffusion-ADNI", 
    entity="s1lentcs",
    config={
        "learning rate": lr,
        "epochs": epochs,
        "optimizer": type(optimizer).__name__,
        "scheduler": type(scheduler).__name__,
        "loss": type(criterion).__name__,
    }
)

model.train()
for epoch in range(epochs):
    total_loss = 0

    loop = tqdm(train_loader_AD, total=len(train_loader_AD))
    for i, real in enumerate(loop):
        real = real.to(device)

        # generate latent space images
        latent = torch.randn(real.shape).to(device)
        generated = model(latent)

        # calculate loss
        loss = criterion(generated, real)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

    scheduler.step(total_loss)
    wandb.log({'loss': total_loss})

    print(f'epoch: {epoch}, loss: {total_loss}')




