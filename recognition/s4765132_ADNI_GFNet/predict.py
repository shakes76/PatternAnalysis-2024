import torch
from modules import GFNet  
from dataset_split import test_loader  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = GFNet(num_classes=2).to(device)
model.load_state_dict(torch.load('alzheimer_gfnet_split.pth'))  
model.eval()  


def predict(model, test_loader):
    correct = 0
    total = 0
    predictions = []

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.append((predicted, labels))

            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy * 100:.2f}%')

    for i, (pred, label) in enumerate(predictions[:5]): 
        print(f'Prediction: {pred.cpu().numpy()}, Ground Truth: {label.cpu().numpy()}')


predict(model, test_loader)

