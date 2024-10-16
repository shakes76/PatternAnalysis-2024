import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import ADNIDataset
from modules import VisionTransformer

def load_model(model_path, device):
    model = VisionTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  
    return model

def predict_and_visualize(model, test_loader, device, num_images=16):
    model.eval()
    images_shown = 0
    fig, axs = plt.subplots(4, 4, figsize=(15, 15)) 
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for i in range(images.size(0)):
                if images_shown >= num_images:
                    break
                
                # Convert the tensor back to image
                img = transforms.ToPILImage()(images[i].cpu())
                true_label = "AD" if labels[i].item() == 1 else "NC"
                predicted_label = "AD" if preds[i].item() == 1 else "NC"
                
                ax = axs[images_shown // 4, images_shown % 4] 
                ax.imshow(img, cmap="gray")
                ax.set_title(f"True: {true_label}, Pred: {predicted_label}")
                ax.axis('off')
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.savefig("predicted_labels.png")

def main():
    data_dir = "/home/groups/comp3710/ADNI/AD_NC"
    model_path = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_path, device)
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    test_dataset = ADNIDataset(root_dir=data_dir, split='test', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    predict_and_visualize(model, test_loader, device, num_images=16)

if __name__ == "__main__":
    main()

