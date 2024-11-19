from torch.utils.data import DataLoader
from dataset import ADNIDataset
import torch
from functools import partial
import torch.nn as nn
from modules import SwinTransformer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import argparse
from sklearn.metrics import confusion_matrix

def accuracy(model, dataloader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Set the model to evaluation
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()
            labels = labels.to(device).long()
            
            preds = model(images)
            
            
            
            _, predicted = torch.max(preds, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

def get_accuracy_from_path(model_path):
    
    model = SwinTransformer(
        img_size=256,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), ape=False, patch_norm=True,
        use_checkpoint=False
    )
    
    model.load_state_dict(torch.load(model_path))
    
    test_ad_path = './test/AD'
    test_nc_path = './test/NC'
    
    test_dataset = ADNIDataset(test_ad_path, test_nc_path, transform=False)
    
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    acc = accuracy(model, test_loader)
    
    return acc

def generate_confusion_matrix(model_path):
        
        model = SwinTransformer(
            img_size=256,
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, qk_scale=None, drop_rate=0, attn_drop_rate=0, drop_path_rate=0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), ape=False, patch_norm=True,
            use_checkpoint=False
        )
        
        model.load_state_dict(torch.load(model_path))
        
        test_ad_path = './test/AD'
        test_nc_path = './test/NC'
        
        test_dataset = ADNIDataset(test_ad_path, test_nc_path, transform=False)
        
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        y_true = []
        y_pred = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device).float()
                labels = labels.to(device).long()
                
                preds = model(images)
                
                _, predicted = torch.max(preds, 1)
                
                y_true += labels.tolist()
                y_pred += predicted.tolist()
                
        return y_true, y_pred

def produce_confusion_matrix(y_true, y_pred, classes):
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
def generate_classification_report(model_path):
        
        y_true, y_pred = generate_confusion_matrix(model_path)
        
        print(classification_report(y_true, y_pred))
        


def main():
    parser = argparse.ArgumentParser(description='Generate confusion matrix and classification report')
    parser.add_argument('model_path', type=str, help='Path to the model')
    parser.add_argument('--confusion_matrix', action='store_true', help='Generate confusion matrix')
    parser.add_argument('--classification_report', action='store_true', help='Generate classification report')
    parser.add_argument('--accuracy', action='store_true', help='Generate accuracy')
    
    args = parser.parse_args()
    
    if args.confusion_matrix:
        y_true, y_pred = generate_confusion_matrix(args.model_path)
        produce_confusion_matrix(y_true, y_pred, ['AD', 'NC'])

    if args.classification_report:
        generate_classification_report(args.model_path)
        
    if args.accuracy:
        acc = get_accuracy_from_path(args.model_path)
        print(f"Accuracy: {acc}")
        
if __name__ == '__main__':
    main()
    