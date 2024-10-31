import torch

class LesionDetectionModel:
    def __init__(self, model_weights='yolov7.pt', device='cpu'):
        """
        Initializes the YOLOv7 model for lesion detection using PyTorch Hub.

        Parameters:
            model_weights (str): Path to the pre-trained YOLOv7 weights.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        
        # Load the model using PyTorch Hub, specifying custom weights if needed
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_weights, source='github')
        self.model.to(self.device)

    def forward(self, images):
        """
        Performs a forward pass through the model.

        Parameters:
            images (torch.Tensor): Batch of images to process.

        Returns:
            torch.Tensor: Model output with predictions for each bounding box.
        """
        with torch.no_grad():
            images = images.to(self.device)
            pred = self.model(images)[0]  # Get predictions
        return pred

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
