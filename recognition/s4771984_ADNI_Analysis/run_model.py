import torch
import argparse
from modules import GFNet
from dataset import get_data_loaders
from train import train_model_with_val_loss, plot_metrics
from predict import evaluate_model, plot_confusion_matrix
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(mode='train', num_epochs=200, lr=0.001, batch_size=64, train_dir='/content/drive/MyDrive/ADNI/AD_NC/train', test_dir='/content/drive/MyDrive/ADNI/AD_NC/test', model_path='saved_model.pth'):
    # Load the data
    train_loader, val_loader, test_loader = get_data_loaders(train_dir, test_dir, batch_size=batch_size)
    
    # Initialize the model
    model = GFNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if mode == 'train':
        # Train the model
        print("Training the model...")
        train_loss, val_loss, train_acc, val_acc = train_model_with_val_loss(
            model, criterion, optimizer, num_epochs, train_loader, val_loader)
        
        # Plot metrics
        plot_metrics(train_loss, val_loss, train_acc, val_acc)
        
        # Save the trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    elif mode == 'evaluate':
        # Load the trained model for evaluation
        model.load_state_dict(torch.load(model_path))
        print("Evaluating the model...")
        all_preds, all_labels, test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        plot_confusion_matrix(all_labels, all_preds)
    
    elif mode == 'predict':
        # Load the model for prediction
        model.load_state_dict(torch.load(model_path))
        print("Running predictions...")
        all_preds, all_labels, _ = evaluate_model(model, test_loader, criterion)
        
        # Display example predictions if desired
        plot_confusion_matrix(all_labels, all_preds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Alzheimer’s disease classification model")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'], default='train', help='Mode to run the script: train, evaluate, or predict')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training dataset directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test dataset directory')
    parser.add_argument('--model_path', type=str, default='saved_model.pth', help='Path to save/load the trained model')

    args = parser.parse_args()
    main(
        mode=args.mode,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        model_path=args.model_path
    )
