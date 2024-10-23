import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import Yolov7
from dataset import DataSetProcessorTrainingVal, DataSetProcessorTest
from torchvision import transforms
from yolov7.utils.loss import ComputeLoss

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
    model.train()
    print("one epoch")
    i = 0
    running_loss = 0
    for image, label in training_dataset:
        # print(image)
        # if image is None or label is None:
        #     print("jjkj")
        #     continue
        print("training loop", i)
        image = image.to(device)
        label = label.to(device)
        # print(image)
        # print(label)
        prediction = model(image)
        #print("Prediction shape:", prediction.shape)
        
        loss, loss_items = loss_function(prediction, label)
        print(loss.requires_grad, "lossss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for this batch
        running_loss = running_loss + loss.item()
        print(running_loss, loss.item())
        i+=1
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(training_dataset)
    print(f"Training Loss: {avg_loss:.4f}")
    
    return avg_loss


def validate():
    model.eval()
    running_loss = 0
    i = 0
    with torch.no_grad():
        for image, label in validation_dataset:
            #print(image, "kk")
            print("Validation loop", i)
            # if image is None or label is None:
            #     print("jjkj")
            #     continue
            image = image.to(device)
            label = label.to(device)
            prediction = model(image)
            #print("Prediction shape:", prediction)
            
            loss, loss_image = loss_function(prediction[1], label)
            running_loss += loss.item()
            i += 1
    avg_val_loss = running_loss / len(validation_dataset)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    return avg_val_loss


for epoch in range(epochs):
    train_one_epoch()
    validate()




