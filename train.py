import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import Yolov7
from dataset import DataSetProcessorTrainingVal, DataSetProcessorTest
from torchvision import transforms

# Varibles
BATCH_SIZE = 16 
epochs = 8
LEARNING_RATE = 0.001

# Paths
train_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1-2_Training_Input_x2'
annotation_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/train_labels'
val_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1_Training_GroundTruth_x2'
test_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1-2_Test_Input'

# function that transforms the data
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image to 128x128
    transforms.ToTensor(),            # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])


# set up Data change ?
train = DataSetProcessorTrainingVal(train_dir, annotation_dir, transform)
validation = DataSetProcessorTrainingVal(val_dir, annotation_dir, transform)
test = DataSetProcessorTest(test_dir, transform)

training_dataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Yolov7()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model = model.to(device)


def train_one_epoch():
    return None


def validate():
    return None


for epoch in range(epochs):
    train_one_epoch()
    validate()




