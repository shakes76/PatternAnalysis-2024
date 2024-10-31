import torch
import torch.nn as nn
from yolov7.models.yolo import Model  
from yolov7.utils.general import non_max_suppression  
from yolov7.utils.torch_utils import select_device, time_synchronized

class LesionDetectionModel:
    def __init__(self, model_weights, device='cuda'):
        """
        Initializes the YOLOv7 model for lesion detection.

        Parameters:
            model_weights (str): Path to the pre-trained YOLOv7 weights.
            device (str): Device to load the model on ('cuda' or 'cpu').
        """
        self.device = select_device(device)
        self.model = Model(cfg='yolov7/cfg/training/yolov7-custom.yaml').to(self.device)
        self.model.load_state_dict(torch.load(model_weights, map_location=self.device)['model'])
        self.model.eval()

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
            pred = self.model(images)[0]  #Get predictions
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

#Helper function to process model output 
def process_detections(detections, img_size):
    """
    Processes the model output to obtain filtered bounding boxes.

    Parameters:
        detections (list of torch.Tensor): Raw model detections.
        img_size (tuple): Size of the image for scaling boxes.

    Returns:
        list of dicts: Processed bounding boxes with class labels and coordinates.
    """
    processed_detections = []
    for det in detections:
        if det is not None:
            det[:, :4] = scale_coords(img_size, det[:, :4])  #Scale boxes
            for *xyxy, conf, cls in det:
                box = {
                    'coordinates': [int(x.item()) for x in xyxy],  #Convert to int
                    'confidence': conf.item(),
                    'class': int(cls.item())
                }
                processed_detections.append(box)
    return processed_detections
