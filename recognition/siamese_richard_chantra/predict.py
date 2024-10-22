import torch
from train import SiameseNetwork
from dataset import preprocess_image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = SiameseNetwork().to(device)

# Load saved model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Function to make a prediction
def predict_similarity(img1, img2, model):
    model.eval() 
    with torch.no_grad(): 
        img1, img2 = img1.to(device), img2.to(device)
        output1, output2 = model(img1, img2)
        distance = torch.sqrt(torch.sum((output1 - output2) ** 2))
    return distance.item()

# Example usage:
if __name__ == "__main__":
    pass