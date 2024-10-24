# containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import Yolov7
from dataset import DataSetProcessorTrainingVal, DataSetProcessorTest
from torchvision import transforms
from yolov7.utils.loss import ComputeLoss
import matplotlib.pyplot as plt

# Varibles
BATCH_SIZE = 16 
epochs = 1
LEARNING_RATE = 0.001

# Paths
train_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1-2_Training_Input_x2'
annotation_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/train_labels'
val_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1_Training_GroundTruth_x2'
test_dir = 'C:/Users/Administrator/Documents/PatternAnalysis-2024/recognition/Data/ISIC2018_Task1-2_Test_Input'

# function that transforms the data
transform_train = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image to 128x128
    transforms.ToTensor(),            # Convert PIL Image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

transform_val = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Ensure these match training
])

# set up Data change ?
train = DataSetProcessorTrainingVal(train_dir, annotation_dir, transform_train)
validation = DataSetProcessorTrainingVal(val_dir, annotation_dir, transform_val)
test = DataSetProcessorTest(test_dir, transform_train)

training_dataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
test_dataset = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Yolov7()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model = model.to(device)

# Define and attach hyperparameters
hyp = {
    'box': 0.05,  # box loss gain
    'cls': 0.5,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive weight
    'obj': 1.0,  # obj loss gain (scale with pixels)
    'obj_pw': 1.0,  # obj BCELoss positive weight
    'iou_t': 0.20,  # IoU training threshold
    'anchor_t': 4.0,  # anchor-multiple threshold
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default gamma is 1.5)
    'hsv_h': 0.015,  # image HSV-Hue augmentation (fraction)
    'hsv_s': 0.7,  # image HSV-Saturation augmentation (fraction)
    'hsv_v': 0.4,  # image HSV-Value augmentation (fraction)
    'degrees': 0.0,  # image rotation (+/- deg)
    'translate': 0.1,  # image translation (+/- fraction)
    'scale': 0.5,  # image scale (+/- gain)
    'shear': 0.0  # image shear (+/- deg)
}

yolo_model = model.model

if hasattr(yolo_model, 'model'):
    yolo_model = yolo_model.model
yolo_model.hyp = hyp
yolo_model.gr = 1.0
loss_function = ComputeLoss(yolo_model)  # Initialize loss

train_loss_list = []
val_loss_list = []
train_accuracy_list = []
val_accuracy_list = []
train_iou_list = []
val_iou_list = []
def train_one_epoch(model, training_dataset, optimizer, loss_function, device):
    model.train()
    print("one epoch")
    i = 0
    running_loss = 0
    total_accuracy = 0
    total_iou = 0
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
        class_predictions = prediction[0]
        bbox_predictions = prediction[1]
        
        # Update accuracy and IoU
        accuracy = calculate_accuracy(class_predictions, label)
        total_accuracy += accuracy
        
        # Example: Assuming label also contains bounding boxes in the same format
        iou = calculate_iou(bbox_predictions, label[:, 1:])
        total_iou += iou
        i+=1
    
    avg_loss = running_loss / i
    avg_accuracy = total_accuracy / i
    avg_iou = total_iou / i
    print("avg_loss: ", avg_loss, "avg_accuracy: ", avg_accuracy, "avg_iou" , avg_iou)
    
    return avg_loss, avg_accuracy, avg_iou

def calculate_accuracy(predictions, labels):
    # Assuming your predictions and labels are in a format where the first dimension is the batch size
    # and the second dimension contains the class probabilities or the class labels
    correct = 0
    total = 0
    
    # Assuming binary classification for simplicity, modify as needed for your use case
    predicted_classes = torch.argmax(predictions, dim=1)
    actual_classes = torch.argmax(labels, dim=1)
    
    correct += (predicted_classes == actual_classes).sum().item()
    total += labels.size(0)  # Count total number of images
    
    accuracy = correct / total
    return accuracy

def calculate_iou(pred_boxes, true_boxes, threshold=0.5):
    # Calculate Intersection Over Union
    # This will need to be adjusted based on your specific bounding box format
    # pred_boxes, true_boxes should be in [x1, y1, x2, y2] format
    
    x1 = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]) * (true_boxes[:, 3] - true_boxes[:, 1])
    
    union = pred_area + true_area - intersection
    iou = intersection / union
    
    return iou.mean().item()

def validate(model, validation_dataset, loss_function, device):
    model.eval()
    running_loss = 0
    total_accuracy = 0
    total_iou = 0
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
             # Calculate accuracy and IoU
            class_predictions = prediction[0]
            bbox_predictions = prediction[1]
            
            accuracy = calculate_accuracy(class_predictions, label)
            total_accuracy += accuracy
            
            iou = calculate_iou(bbox_predictions, label[:, 1:])
            total_iou += iou
            i += 1
    avg_val_loss = running_loss / i
    avg_accuracy = total_accuracy / i
    avg_iou = total_iou / i
    print("avg_val_loss: ", avg_val_loss, "avg_accuracy: ", avg_accuracy, "avg_iou" , avg_iou)
    return avg_val_loss, avg_accuracy, avg_iou

def save(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

# Training loop

for epoch in range(epochs):
    train_loss, train_accuracy, train_iou = train_one_epoch(model, training_dataset, optimizer, loss_function, device)
    print(train_loss)
    
    validate_loss, val_accuracy, val_iou = validate(model, validation_dataset, loss_function, device)
    print(validate_loss)

    train_loss_list.append(train_loss)
    val_loss_list.append(validate_loss)
    train_accuracy_list.append(train_accuracy)
    val_accuracy_list.append(val_accuracy)
    train_iou_list .append(train_iou)
    val_iou_list.append(val_iou)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train IoU: {train_iou:.4f}")
    print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {validate_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val IoU: {val_iou:.4f}")
    save(model, optimizer, epoch, f'checkpoint_epoch_{epoch+1}.pth')


plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()


plt.savefig('loss_plot.png')
plt.close()  