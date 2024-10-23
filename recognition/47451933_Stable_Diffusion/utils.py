import torch
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

class NoiseScheduler:
    '''
        class to determin the what level of noise
        to apply at each timestep
    '''
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.1):
        '''
            beta_start and beta_end: determin how much noise will be added
            nume_timestep: number of times steps to use
        '''
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

    def get_beta(self, t):
        '''
            returns bata at timestep t
        '''
        return self.betas[t]

    def get_alphas(self):
        '''
            alphas -> 1-betas
        '''
        return 1 - self.betas

    def get_alphas_cumprod(self):
        '''
            cumulaitve product of alphas
            used to calculate how much noise to add
        '''
        return torch.cumprod(self.get_alphas(), dim=0)
    
def add_noise(x, t, nosie_schedular):
    noise = torch.randn_like(x).to(device)
    alpha_t = nosie_schedular.get_alphas_cumprod()[t].to(x.device)
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

def reverse_diffusion(noisy_latent, predicted_noise, t, nosie_schedular):
    '''
        reversese the noise in noisy_letent at timestep t using the predicted noise
    '''
    
    # get the details about the noise at timestep t from the nosie_schedular
    alpha_t = nosie_schedular.get_alphas_cumprod()[t].to(noisy_latent.device)
    beta_t = nosie_schedular.get_beta(t).to(noisy_latent.device)
    alpha_prev = nosie_schedular.get_alphas_cumprod()[t - 1].to(noisy_latent.device) if t > 0 else torch.tensor(1.0).to(noisy_latent.device)

    # Predicted noise should be subtracted
    mean = (noisy_latent - beta_t * predicted_noise / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t)

    if t > 0:
        variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
        z = torch.randn_like(noisy_latent)
        return mean + torch.sqrt(variance) * z
    else:
        return mean