import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import SiameseNetwork, ContrastiveLoss
from dataset import get_pair_data_loader
import os

num_epochs = 20
learning_rate = 0.001
batch_size = 32

csv_file = 'data/ISIC_2020_Training_GroundTruth_v2.csv'
img_dir = 'data/train/'


def main():
    train_loader = get_pair_data_loader(csv_file, img_dir, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"CUDA Check: {torch.cuda.is_available()}")
    model = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]", end="")
        epoch_loss = 0.0
        model.train()
        for batch_idx, (img1, img2, label) in enumerate(train_loader):
            img1, img2, label = img1.cuda(), img2.cuda(), label.float().cuda()
            optimizer.zero_grad()

            outputs = model(img1, img2)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        print(f", Loss: {avg_epoch_loss:.4f}")

    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    os.makedirs('models', exist_ok=True)
    model_path = 'models/siamese_network.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')


if __name__ == "__main__":
    main()
