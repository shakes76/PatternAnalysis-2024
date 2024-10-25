import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import os
from modules import VQVAE, PixelCNN, GatedPixelCNN
import dataset
from skimage.metrics import structural_similarity as ssim

from tqdm import tqdm

def readable_timestamp():
    """
    Timestamp for model naming
    """
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def train(args, train_loader, val_loader, optimizer, model):
    """
    Trains VQVAE model.
    param train_loader: loaded training dataset
    param val_loader: loaded validation dataset 
    param optimizer : the optimiser use 
    param: the model to be used

    """
    results = {
    'n_updates': 0,
    'n_val_updates': [],
    'train_recon_errors': [],
    'val_recon_errors': [],
    'train_loss': [],
    'val_loss': [],
    'val_ssim': [],
    'train_perplexities': [],
    'val_perplexities': []
    }

    # Train vqvae model
    for i in range(args.n_updates):
        # (x, _) = next(iter(training_loader))    # b, c, h, w
        model.train()
        x = next(iter(train_loader))
        x = x.to(args.device)
        optimizer.zero_grad()


        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["train_recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["train_perplexities"].append(perplexity.cpu().detach().numpy())
        results["train_loss"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i
        # validate model
        if i % args.log_interval == 0:
            ### TODO: validate
            model.eval()
            with torch.no_grad():
                n_b = 0
                ssim_sum = 0
                recon_loss_avg = 0
                val_loss_avg = 0
                val_perplexities = 0
                for x in val_loader:
                    x = x.to(args.device)
                    embedding_loss, x_hat, perplexity = model(x)
                    recon_loss = torch.mean((x_hat - x)**2)
                    loss = recon_loss + embedding_loss
                    # print(x.shape)
                    # print(x_hat.shape)
                    ind_score = [ssim(a.cpu().squeeze().numpy(), b.squeeze().detach().cpu().numpy(), data_range=-1.0) for a, b in zip(x, x_hat)]
                    n_b += len(x)
                    ssim_sum += sum(ind_score)
                    recon_loss_avg += recon_loss.item()
                    val_loss_avg += loss.item()
                    val_perplexities += perplexity
                ssim_avg = ssim_sum / n_b
                recon_loss_avg /= len(val_loader)
                val_loss_avg /= len(val_loader)
                val_perplexities /= len(val_loader)
                results["val_recon_errors"].append(recon_loss_avg)
                results["val_loss"].append(val_loss_avg)
                results['val_ssim'].append(ssim_avg)
                results["n_val_updates"].append(i)
                results["val_perplexities"].append(val_perplexities)
                    
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                dataset.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["train_recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["train_loss"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["train_perplexities"][-args.log_interval:]))
            

def train_pixel_cnn(args):
    """
    Trains the pixel cnn model
    REFERENCE:https://github.com/anordertoreclaim/PixelCNN/tree/master/pixelcnn
    """
    train_loader, test_loader = dataset.load_latent_block(batch_size=8)
    args.img_dim1 = 32
    args.img_dim2 = 32
    args.n_layers = 10
    model = GatedPixelCNN(args.n_embeddings, args.img_dim1 * args.img_dim2, args.n_layers).to(args.device)

    checkpoint_path = './pixel_cnn_reshape2.pt'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Model loaded from", checkpoint_path)
    criterion = nn.CrossEntropyLoss().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 50):
        # train
        model.train()
        train_loss = []
        for batch_idx, (x, label) in tqdm(enumerate(train_loader)):
            start_time = time.time()
            x = (x[:, 0]).to(args.device)
            # x = x.cuda()
            label = label.to(args.device)
            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()

            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss.append(loss.item())
        
        # validate
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_idx, (x, label) in tqdm(enumerate(test_loader)):
                x = (x[:, 0]).to(args.device)
                label = label.to(args.device)
                logits = model(x, label)
                logits = logits.permute(0, 2, 3, 1).contiguous()

                loss = criterion(
                    logits.view(-1, args.n_embeddings),
                    x.view(-1)
                )
                val_loss.append(loss.item())
        
        print(f'Epoch {epoch}: train loss is {sum(train_loss) / len(train_loader)}, validation_loss is {sum(val_loss) / len(test_loader)}')
        torch.save(model.state_dict(), './pixel_cnn_reshape2.pt')


def main(args):

    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if args.save:
        print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')
    #Load data and define batch data loaders

    training_data, validation_data, training_loader, validation_loader = dataset.load_dataset(args)
    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(args.device)

    #Set up optimizer and training loop
 
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    if args.mode == 'vq_vae':
        train(args, training_loader, validation_loader, optimizer, model)
    elif args.mode == 'pixel_cnn':
        train_pixel_cnn(args)







if __name__ == "__main__":
    timestamp = readable_timestamp()
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--dataset",  type=str, default='mri')
    parser.add_argument('--train_dir', type=str, default='/Selena/Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_train')
    parser.add_argument('--test_dir', type=str, default='/Selena/Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_test')
    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename",  type=str, default=timestamp)

    parser.add_argument('--mode', type=str, choices=['vq_vae', 'pixel_cnn'], default='pixel_cnn')

    args = parser.parse_args()
    main(args)
