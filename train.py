from dataset import load_data
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import utils
from modules import VQVAE
from predict import generate_samples

# Hyperparameters
HIDDEN_DIM = 128
RES_HIDDEN_DIM = 32
N_RES_LAYERS = 5
N_EMBEDDINGS = 512
EMBEDDING_DIM = 64
LEARNING_RATE = 2e-4
N_EPOCHS = 200

# Metrics
ssim_scores = []
train_losses = []
best_epoch = 0

# Initialize model and optimisers
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
train_loader, test_loader, val_loader = load_data()
model = VQVAE(HIDDEN_DIM, 
              RES_HIDDEN_DIM, 
              N_RES_LAYERS, 
              N_EMBEDDINGS, 
              EMBEDDING_DIM).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
criterion = torch.nn.MSELoss()

utils.folder_check()

def train_model():
    for epoch_idx in range(N_EPOCHS):
        print("Training")

        epoch_loss = 0
        epoch_start = time.time()

        model.train()

        for batch, im in enumerate(tqdm(train_loader)):
            start_time = time.time()

            im = im.float().unsqueeze(1).to(device)
            opt.zero_grad()
            
            decoded_output, embedding_loss, encoded_output, quantised_output = model(im)
            recon_loss = criterion(decoded_output, im)
            loss = recon_loss + embedding_loss
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            
            if (batch + 1) % 64 == 0:
                print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                    batch * len(im), len(train_loader.dataset),
                    50 * batch / len(train_loader),
                    epoch_loss/batch,
                    time.time() - start_time
                ))
        
        if (epoch_idx + 1) % 10 == 0:
            print("Generating Epoch Image")
            generate_samples(test_loader, model, epoch_idx+1)
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print('Finished epoch {} in time: {} with loss:'.format(
            epoch_idx + 1, epoch_start - time.time(), avg_epoch_loss))
        
        validate_model(epoch_idx+1)
        
        scheduler.step()
        
    print('Done Training...')


def plot_results():
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_EPOCHS + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('outputs/training_loss.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_EPOCHS + 1), ssim_scores, label='SSIM Scores')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Score')
    plt.title('SSIM Scores over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('outputs/ssim_scores.png')
    plt.close()
    

def validate_model(epoch):
    print("Validating")
    global best_epoch
    model.eval()
    total_ssim = 0
    
    with torch.no_grad():
        for batch, im in enumerate(val_loader):
            im = im.float().unsqueeze(1).to(device)
            
            decoded_output, _, _, _ = model(im)
            
            total_ssim += utils.calc_ssim(decoded_output, im)
        
    epoch_ssim_score = total_ssim/(batch+1)
    ssim_scores.append(epoch_ssim_score)
    print(max(ssim_scores), "best epoch")
    print(best_epoch)
    print(epoch_ssim_score, "epoch score")
    print(epoch)
    if epoch_ssim_score == max(ssim_scores):
        torch.save(model.state_dict(), f'models/checkpoint_epoch{epoch}_vqvae.pt')
        best_epoch = epoch
        print(f"Achieved an SSIM score of {epoch_ssim_score}, NEW BEST! saving model")
    else:
        print(f"Achieved an SSIM score of {epoch_ssim_score}")


def test():
    print("Testing")
    model.load_state_dict(torch.load(f'models/checkpoint_epoch{best_epoch}_vqvae.pt'))
    torch.save(model.state_dict(), f'outputs/final_vqvae.pt')
    model.eval()
    total_ssim = 0
    
    with torch.no_grad():
        for batch, im in enumerate(test_loader):
            im = im.float().unsqueeze(1).to(device)
            
            decoded_output, _, _, _ = model(im)
            
            total_ssim += utils.calc_ssim(decoded_output, im)
        
    total_test_ssim = total_ssim/(batch + 1)
    return total_test_ssim


if __name__ == "__main__":   
    start = time.time()
    train_model()
    plot_results()
    print(f"Took {(time.time() - start) / 60} minutes to train")
    test_ssim = test()
    print(f"Test SSIM achieved as {test_ssim}")
    generate_samples(test_loader, model)
