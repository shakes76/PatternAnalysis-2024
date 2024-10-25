import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from modules import dice_coeff, multiclass_dice_coeff, UNet, CombinedDataset, dice_loss
from dataset import load_data_2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import wandb

dir_img = Path('C:/Users/rober/Desktop/COMP3710/keras_slices_test')
dir_mask = Path('C:/Users/rober/Desktop/COMP3710/keras_slices_seg_test')
dir_img_val = Path('C:/Users/rober/Desktop/COMP3710/keras_slices_validate')
dir_mask_val = Path('C:/Users/rober/Desktop/COMP3710/keras_slices_seg_validate')
dir_checkpoint = Path('./checkpoints')

batch_losses = []
val_dice_scores = []
conf_matrix_total = None

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device_type = 'cuda'):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long).squeeze(1)
            mask_true = torch.clamp(mask_true, min=0, max=1)

            mask_pred = net(image)

            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # Confusion matrix for binary classification
                # preds_flat = mask_pred.view(-1).cpu().numpy()
                # labels_flat = mask_true.view(-1).cpu().numpy()
                # conf_matrix = confusion_matrix(labels_flat, preds_flat)

            else:
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # Confusion matrix for multi-class classification
                # preds_flat = mask_pred.argmax(dim=1).view(-1).cpu().numpy()
                # labels_flat = mask_true.argmax(dim=1).view(-1).cpu().numpy()
                # conf_matrix = confusion_matrix(labels_flat, preds_flat)

            # if conf_matrix_total is None:
            #     conf_matrix_total = conf_matrix
            # else:
            #     conf_matrix_total += conf_matrix

    net.train()
    
    return dice_score / max(num_val_batches, 1)

def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    image_files = [os.path.join(dir_img, f) for f in os.listdir(dir_img) if f.endswith('.nii.gz') or f.endswith('.nii')]
    # Step 2: Load the images using load_data_2D
    images = load_data_2D(image_files, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False)

    image_files_mask = [os.path.join(dir_mask, f) for f in os.listdir(dir_mask) if f.endswith('.nii.gz') or f.endswith('.nii')]
    # Step 2: Load the images using load_data_2D
    images_mask = load_data_2D(image_files_mask, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False)

    image_files_val = [os.path.join(dir_img_val, f) for f in os.listdir(dir_img_val) if f.endswith('.nii.gz') or f.endswith('.nii')]
    # Step 2: Load the images using load_data_2D
    images_val = load_data_2D(image_files_val, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False)

    image_files_mask_val = [os.path.join(dir_mask_val, f) for f in os.listdir(dir_mask_val) if f.endswith('.nii.gz') or f.endswith('.nii')]
    # Step 2: Load the images using load_data_2D
    images_mask_val = load_data_2D(image_files_mask_val, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False)


    training_set = CombinedDataset(images, images_mask)
    validate_set = CombinedDataset(images_val, images_mask_val)
    # 2. Split into train / validation partitions
    n_val = int(len(images) * val_percent)
    n_train = len(images) - n_val
    train_set, val_set = training_set, validate_set
    print(len(train_set))

    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True)

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images, true_masks = batch
                #print("masks ", true_masks.size())
                true_masks = true_masks.squeeze(1)
                true_masks = torch.clamp(true_masks, min=0, max=1)

                #print("hello: ", images.size())

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     'train loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                batch_losses.append(loss.item())
                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        val_dice_scores.append(val_score)
                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            pass
                            # experiment.log({
                            #     'learning rate': optimizer.param_groups[0]['lr'],
                            #     'validation Dice': val_score,
                            #     'images': wandb.Image(images[0].cpu()),
                            #     'masks': {
                            #         'true': wandb.Image(true_masks[0].float().cpu()),
                            #         'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            #     },
                            #     'step': global_step,
                            #     'epoch': epoch,
                            #     **histograms
                            # })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = training_set.image_masks
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    plt.figure(figsize=(10, 5))
    plt.plot(batch_losses, label='Batch Loss')
    plt.title('Batch Loss During Training')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    val_dice_scores_cpu = [score.cpu().item() for score in val_dice_scores]
    # Plot validation Dice scores
    plt.figure(figsize=(10, 5))
    plt.plot(val_dice_scores_cpu, label='Validation Dice Score')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.show()

    # # Confusion Matrix Plot
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upksampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

args = get_args()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')

# Change here to adapt to your data
# n_channels=3 for RGB images
# n_classes is the number of probabilities you want to get per pixel
model = UNet(n_channels=1, n_classes=2, bilinear=args.bilinear)
print("hi: ", args.classes)
model = model.to(memory_format=torch.channels_last)

logging.info(f'Network:\n'
              f'\t{model.n_channels} input channels\n'
              f'\t{model.n_classes} output channels (classes)\n'
              f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

# if args.load:
#     state_dict = torch.load(args.load, map_location=device)
#     del state_dict['mask_values']
#     model.load_state_dict(state_dict)
#     logging.info(f'Model loaded from {args.load}')

model.to(device=device)
try:
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )
except torch.cuda.OutOfMemoryError:
    logging.error('Detected OutOfMemoryError! '
                  'Enabling checkpointing to reduce memory usage, but this slows down training. '
                  'Consider enabling AMP (--amp) for fast and memory efficient training')
    torch.cuda.empty_cache()
    model.use_checkpointing()
    train_model(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_scale=args.scale,
        val_percent=args.val / 100,
        amp=args.amp
    )

