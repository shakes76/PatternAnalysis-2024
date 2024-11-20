import torch
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from dataset import getDataLoader
from modules import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32

# Data transformations
data_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def predict(model, test_loader):
    # Initialize evaluation variables
    model.eval()
    total = 0
    correct = 0
    label_list = []
    prediction_list = []

    # Progress bar for visualization
    progress_bar = tqdm(test_loader, desc="Predicting Labels on Test Set")

    # Perform model evaluation based on the test set
    with torch.no_grad():
        for img1, img2, labels in progress_bar:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            # Forward pass
            outputs = model(img1, img2).squeeze()
            predictions = (outputs > 0.5).float()

            # Store predictions and labels for metric calculations
            label_list.extend(labels.cpu().numpy())
            prediction_list.extend(predictions.cpu().numpy())

            # Calculate accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    # Calculate evaluation scores
    accuracy = correct / total
    precision = precision_score(label_list, prediction_list)
    recall = recall_score(label_list, prediction_list)
    f1 = f1_score(label_list, prediction_list)

    print(f"\nTest Accuracy: {accuracy * 100:.2f}")
    print(f"Precision: {precision * 100:.2f}")
    print(f"Recall: {recall * 100:.2f}")
    print(f"F1-score: {f1 * 100:.2f}")

    # Show confusion matrix
    conf_matrix = confusion_matrix(label_list, prediction_list)
    print("\nConfusion Matrix:")
    print(conf_matrix)


# Load test dataset
test_loader = getDataLoader(data_transforms, batch_size, method="mixed", train=False)

# Load model
model = SiameseNetwork().to(device)

model_path = "siamese_model_final.pth"
model.load_state_dict(torch.load(model_path))

# Predict model output on test set
predict(model, test_loader)
