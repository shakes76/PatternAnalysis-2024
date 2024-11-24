import torch
import torch.nn as nn

class LesionDetectionModel(nn.Module):
    def __init__(self, model_weights='yolov7.pt', device='cpu'):
        """
        Initializes the YOLOv7 model for lesion detection using PyTorch Hub with additional dropout layers.

        Parameters:
            model_weights (str): Path to the pre-trained YOLOv7 weights.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        super(LesionDetectionModel, self).__init__()

        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')

        # Load the YOLO model without the autoShape wrapper to get direct access to its layers
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_weights, source='github', autoshape=False)
        self.model.to(self.device)

        # Attempt to freeze backbone layers if they exist in the model
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Add dropout after certain layers
        self.dropout = nn.Dropout(p=0.2)  # Example of a dropout layer with 50% probability

    def forward(self, images):
        """
        Performs a forward pass through the model.

        Parameters:
            images (torch.Tensor): Batch of images to process.

        Returns:
            torch.Tensor: Model output with predictions for each bounding box.
        """
        images = images.to(self.device)

        # Perform a forward pass through the original model
        x = self.model(images)[0]

        # Apply dropout before returning output
        x = self.dropout(x)

        return x

    def detect(self, images, conf_thres=0.25, iou_thres=0.8):
        """
        Runs detection on input images with specified thresholds.

        Parameters:
            images (torch.Tensor): Batch of images to process.
            conf_thres (float): Confidence threshold for predictions.
            iou_thres (float): IoU threshold for non-max suppression.

        Returns:
            list of torch.Tensor: Bounding boxes and labels for detected lesions.
        """
        pred = self.forward(images)
        detections = non_max_suppression(pred, conf_thres, iou_thres)
        return detections
