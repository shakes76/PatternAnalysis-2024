import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import  matplotlib.pyplot as plt

from tqdm import *
from modules import *
from dataset import BrainDataset
from ema_pytorch import EMA
from torch.utils.data import DataLoader


class DDPM(nn.Module):
    def __init__(self, model, device, timestep=1000):
        super().__init__()
        self.model = model
        self.device = device
        self.beta = torch.linspace(1e-4, 0.02, timestep, device=device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alphas_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value = 1.)
        
        snr = self.alpha_bar / (1 - self.alpha_bar)
        self.loss_w = snr / (snr + 1)
    
    def predict_v(self, x_start, t, noise):
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * noise - one_minus_alpha_bar * x_start
    
    def predict_start_from_v(self, x_t, t, v):
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * x_t - one_minus_alpha_bar * v
    
    def predict_noise_from_start(self, x_t, t, x0):
        sqrt_recip = torch.sqrt(1. / self.alpha_bar[t])[:, None, None, None]
        sqrt_recipm1 = torch.sqrt(1. / self.alpha_bar[t] - 1)[:, None, None, None]
        return (sqrt_recip * x_t - x0) / sqrt_recipm1
        
    def forward(self, img, t, noise):
        # xt = x0 * alpha_bar_sqrt + one_minus_alpha_bar * noise 
        alpha_bar = self.alpha_bar[t][:, None, None, None]
        
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * img + one_minus_alpha_bar * noise
    
    def reverse(self, x_t, pred_noise, t):
        x0 = self.predict_start_from_v(x_t, t, pred_noise)

        coef1 = self.beta[t] * self.alphas_bar_prev[t]**0.5 / (1. - self.alpha_bar[t])
        coef2 = (1. - self.alphas_bar_prev[t]) * torch.sqrt(self.alpha[t]) / (1. - self.alpha_bar[t])
        posterior_variance = self.beta[t] * (1. - self.alphas_bar_prev[t]) / (1. - self.alpha_bar[t])

        mean = x0 * coef1[:, None, None, None] + x_t * coef2[:, None, None, None]
        var = torch.log(posterior_variance.clamp(min =1e-20))[:, None, None, None]
        '''
        mean = (1 / (alpha_t ** 0.5)) * (
          x_t - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * pred_noise
        )'''
        
        noise = torch.randn_like(x_t) if t[0] > 0 else 0.
        pred_img = mean + (0.5 * var).exp() * noise
        return pred_img, x0
    
class Trainer():
    def __init__(self, model, dataloader, diff_method, ckpt_dir, load_path=None, \
                 total_step=100000, save_n_step=10000, lr=1e-4, timestep=1000):
        
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs('./results/', exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = model
        self.timestep = timestep
        self.dataloader = dataloader
        self.ckpt_dir = ckpt_dir
        
        self.step = 1
        self.mean_ema_loss = 1
        self.total_step = total_step
        self.save_n_step = save_n_step
        self.ema = EMA(self.model, beta = 0.995, update_every = 2)
        self.ema.to(self.device)
        
        self.diff_method = diff_method(self.model, self.device, timestep=timestep)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler(self.device)
        self.loss_fn = nn.MSELoss(reduction='none')
        #self.loss_fn = nn.L1Loss() 
        
        self.loss_history = []  # List to store loss values
        
        if load_path is not None:
            self.load_state_dict(load_path)
            self.ema.copy_params_from_ema_to_model()
            print("sucessful load state dict !!!!!!")
            print(f"start from step {self.step}")
    
    def state_dict(self, step):
        return {
            "step": step,
            "ema": self.ema.state_dict(),
        }
    
    def load_state_dict(self, path):
        state_dict = torch.load(path)
        self.ema.load_state_dict(state_dict['ema'])
        self.step = state_dict['step']

    def train(self, generate_output = True):
        start = time.time()
        print(f'Start of step {self.step}')
        
        for step in tqdm(range(self.step, self.total_step+1), desc=f"Training progress"):
            self.optimizer.zero_grad()
            img, label = next(iter(self.dataloader))
            img = img.to(self.device)
            label = torch.tensor(label, device = self.device)
            noise = torch.randn_like(img)
            
            t = torch.randint(0, self.timestep, (img.shape[0],), device=self.device)
            noised_img = self.diff_method(img, t, noise)
            target_v = self.diff_method.predict_v(img, t, noise)
            predict_v = self.model(noised_img, t, label)
            loss_w = self.diff_method.loss_w.gather(-1, t)

            loss = self.loss_fn(target_v, predict_v).mean(dim=(1, 2, 3))
            loss = loss * loss_w
            loss = loss.mean()
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1e9)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.ema.update()
            
            self.loss_history.append(loss.item())
        
            if step % self.save_n_step == 0:
                #clear_output(wait=True)
                epoch = step // self.save_n_step
                time_minutes = (time.time() - start) / 60
                if step > (self.total_step // 2):
                    torch.save(self.state_dict(step), f"{self.ckpt_dir}/weight_epoch{epoch}.pt")
                
                print(f"epoch: {epoch}, loss: {loss.data} ~~~~~~")
                print (f'Time taken for epoch {epoch} is {time_minutes:.3f} min\n') 
                print(f"sucessful saving epoch {epoch} state dict !!!!!!!")
                start = time.time()
                
                if generate_output:
                    _ = self.generate(epoch)
                self.ema.copy_params_from_ema_to_model()
        print("finish training: ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss over Steps')
        plt.legend()
        plt.savefig(f'./training_loss.jpg')
        plt.show()
    
    @torch.inference_mode()
    def ddim_sample(self, cond = 'NC'):
        batch, device, total_timesteps, sampling_timesteps, eta = 32, self.device, 1000, 200, 0
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)  
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        img = torch.randn([batch, 1, 128, 128], device = device)

        if cond == 'NC':
            label = torch.zeros(batch).to(self.device)
        else:
            label = torch.ones(batch).to(self.device)
            
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'Sampling loop time step'):
            t = torch.full((batch,), time, device = device, dtype = torch.long)
            pred = self.ema(img, t, label)
            x_start = self.diff_method.predict_start_from_v(img, t, pred)
            pred_noise = self.diff_method.predict_noise_from_start(img, t, x_start)
            
            if time_next < 0:
                img = x_start
                continue

            alpha = self.diff_method.alpha_bar[time]
            alpha_next = self.diff_method.alpha_bar[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        fake_img = img
        num_rows = 4
        num_columns = 8
        
        fig, axs = plt.subplots(num_rows, num_columns, figsize=(16, 8))
        for i in range(num_rows):
            for j in range(num_columns):
                ax = axs[i, j]
                index = i * num_columns + j
                img = (fake_img[index] + 1) / 2
                img = img.clamp(0, 1)

                # Display the image
                ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
                ax.axis('off')
        plt.savefig(f'./{cond}_result_img.jpg')
        plt.show()
        
    def generate(self, epoch, plot_img=True, noise=None):
        self.ema.eval()
        self.fake_imgs = []
        x = torch.randn([32, 1, 128, 128], device=self.device)
        label = torch.cat((torch.ones(16), torch.zeros(16))).to(self.device)
        
        if noise is not None:
            x = noise
            
        with torch.no_grad():
            for t in reversed(range(0, 1000)):
                t_tensor = torch.full((x.size(0),), t, dtype=torch.long, device=self.device)
                pred_noise = self.ema(x, t_tensor, label)
                x, x_start = self.diff_method.reverse(x, pred_noise, t_tensor)
                self.fake_imgs.append(x)
                
        fake_img = x
        num_rows = 4
        num_columns = 8
        
        if plot_img:
            fig, axs = plt.subplots(num_rows, num_columns, figsize=(16, 8))
            for i in range(num_rows):
                for j in range(num_columns):
                    ax = axs[i, j]
                    index = i * num_columns + j
                    img = (fake_img[index] + 1) / 2
                    img = img.clamp(0, 1)

                    # Display the image
                    ax.imshow(img.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
                    ax.axis('off')
            plt.savefig(f'./results/result_img{epoch}.jpg')
            plt.show()
        self.ema.train()
        return self.fake_imgs
    
def main():
    # hyperparameter
    LR = 7e-5
    BATCH_SIZE = 8
    IMG_SIZE = 128
    FILTER_SIZE = 64
    TOTAL_ITERATION = 100000
    SAVE_N_ITERATION = 10000
    CKPT_DIR = './model_weight/'

    # Put the data dir here
    # BrainDataset will resize the image into 128 * 128
    data_dir = '/kaggle/input/braindata-condgen/AD_NC/train'
    dataset = BrainDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training & plot image
    # model will load the weight if load_path is not None
    load_path = None
    model = U_net(FILTER_SIZE)
    trainer = Trainer(model, dataloader, DDPM, CKPT_DIR, load_path=load_path, 
                  total_step=TOTAL_ITERATION, save_n_step=SAVE_N_ITERATION, lr=LR)
    
    trainer.train()
    trainer.plot_loss()

if __name__ == '__main__':
    main()
