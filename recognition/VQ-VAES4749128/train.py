import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from modules import VQVAE, GatedPixelCNN
import dataset
from skimage.metrics import structural_similarity as ssim

def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()

def train(args, train_loader, val_loader, optimizer, model):
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


    for i in range(args.n_updates):
        # (x, _) = next(iter(training_loader))    # b, c, h, w
        model.train()
        x = next(iter(train_loader))
        x = x.to(args.device)
        optimizer.zero_grad()

        # print(x.shape)

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()

        results["train_recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["train_perplexities"].append(perplexity.cpu().detach().numpy())
        results["train_loss"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

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
                    

            # iterate over validation dataset / loader
                # forward to model and get prediction
                # compute similarity
                # validation loss
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









def main(args):
    """
    Hyperparameters
    """
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.save:
        print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

    """
    Load data and define batch data loaders
    """


    training_data, validation_data, training_loader, validation_loader = dataset.load_dataset(args)
    """
    Set up VQ-VAE model with components defined in ./models/ folder
    """

    model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
                args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(args.device)

    """
    Set up optimizer and training loop
    """
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

    train(args, training_loader, validation_loader, optimizer, model)







if __name__ == "__main__":
    timestamp = readable_timestamp()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='/home/zijian/Selena/Vq_vae_comp3710/results/vqvae_data_thu_oct_24_16_06_56_2024.pth')
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
    parser.add_argument('--train_dir', type=str, default='Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_train')
    parser.add_argument('--test_dir', type=str, default='Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_test')
    # whether or not to save model
    parser.add_argument("-save", action="store_true")
    parser.add_argument("--filename",  type=str, default=timestamp)

    args = parser.parse_args()
    main(args)
