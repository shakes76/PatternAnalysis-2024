import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import Yolov7
from dataset import DataSetProcessorTrainingVal, DataSetProcessorTest
from torchvision import transforms
from yolov7.utils.loss import ComputeLoss
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np



class Train:
    
    def __init__(self):
        # Varibles
        BATCH_SIZE = 32
        self.epochs = 8
        LEARNING_RATE = 0.001

        # Paths
        train_dir = 'recognition/Data/ISIC2018_Task1-2_Training_Input_x2'
        annotation_dir = 'recognition/Data/train_labels'
        val_dir = 'recognition/Data/ISIC2018_Task1_Training_GroundTruth_x2'
        test_dir = 'recognition/Data/ISIC2018_Task1-2_Test_Input'
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

        # set up Data change
        train = DataSetProcessorTrainingVal(train_dir, annotation_dir, transform_train)
        validation = DataSetProcessorTrainingVal(val_dir, annotation_dir, transform_val)
        test = DataSetProcessorTest(test_dir, transform_train)

        self.training_dataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        self.validation_dataset = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
        self.test_dataset = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Yolov7()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.model = model.to(self.device)

        # Define and attach hyperparameters
        hyp = {
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0
        }

        yolo_model = model.model

        if hasattr(yolo_model, 'model'):
            yolo_model = yolo_model.model
        yolo_model.hyp = hyp
        yolo_model.gr = 1.0
        self.loss_function = ComputeLoss(yolo_model)

    def train_one_epoch(self, model, training_dataset, optimizer, loss_function, device):
        model.train()
        print("one epoch")
        i = 0
        running_loss = 0
        for image, label in training_dataset:
            print("training loop", i)
            image = image.to(device)
            label = label.to(device)

            prediction = model(image)
            
            loss, _, _= loss_function(prediction, label)
            print(loss.requires_grad, "lossss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for this batch
            running_loss = running_loss + loss.item()     
            i+=1
        
        avg_loss = running_loss / i

        return avg_loss

    def calculate_accuracy(self, iou_values, threshold=0.8):
        if not isinstance(iou_values, torch.Tensor):
            iou_values = torch.tensor(iou_values)
        correct_predictions = (iou_values >= threshold).sum().item()
        total_predictions = len(iou_values)

        accuracy = correct_predictions / total_predictions * 100
        return accuracy

    def load_checkpoint(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def evaluate(self, checkpoint_path='model_checkpoint.pth'):
        self.model.eval()
        iou_scores = []
        i = 0
        try:
            self.load_checkpoint(self.model, self.optimizer, checkpoint_path)
            print("Checkpoint loaded.")
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}. Please train the model first.")
            return
        

        with torch.no_grad():
            for images, labels in self.validation_dataset:
                print("Validation loop", i)
                images = images.to(self.device)
                label = labels.to(self.device)
                prediction = self.model(images)
        
                _, _, IOU = self.loss_function(prediction[1], label)
                average_iou = IOU.mean().item()
                iou_scores.append(average_iou)
                i+=1

        average_iou = np.mean(iou_scores)
        accuracy = self.calculate_accuracy(iou_scores) if iou_scores else 0
        print(f"Average IoU: {average_iou:.4f}, Accuracy: {accuracy:.4f}")
        return average_iou, accuracy


    def save(self, model, optimizer, epoch, path):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, path)

    def start_training(self):
        train_loss_list = []
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(self.model, self.training_dataset, self.optimizer, self.loss_function, self.device)
            print(train_loss)
            train_loss_list.append(train_loss)
            self.save(self.model, self.optimizer, epoch, 'model_checkpoint.pth')
        return train_loss_list, self.model, self.validation_dataset, self.device

        


