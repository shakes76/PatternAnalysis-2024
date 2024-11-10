# Import YOLO from ultralytics package
# Ultralytics YOLO is an updated, Python-native implementation
from ultralytics import YOLO

class YOLOSegmentation:
    """
    A wrapper class for YOLO segmentation model that provides simplified interface
    for training, evaluation, and prediction tasks.
    
    This class encapsulates common YOLO operations and provides a clean API
    for the main tasks in computer vision: training, evaluation, and inference.
    """
    
    def __init__(self, weights_path):
        """
        Initialize the YOLO model with specified weights.
        
        Args:
            weights_path (str): Path to the model weights file.
                              Can be either pretrained weights (e.g., 'yolov8n-seg.pt')
                              or custom trained weights
        """
        # Create YOLO model instance using provided weights
        self.model = YOLO(weights_path)

    def train(self, params):
        """
        Train the YOLO model with given parameters.
        
        Args:
            params (dict): Dictionary containing training parameters such as:
                         - epochs: number of training epochs
                         - batch_size: batch size for training
                         - data: path to data configuration file
                         - imgsz: input image size
                         And other training configurations
        
        Returns:
            results: Training results and metrics
        """
        # Unpack parameters dictionary and train the model
        results = self.model.train(**params)
        return results

    def evaluate(self):
        """
        Evaluate the model on validation dataset.
        
        This method runs validation on the dataset specified
        in the data configuration file used during training.
        
        Returns:
            results: Validation metrics including mAP, precision, recall
        """
        # Run validation and return metrics
        results = self.model.val()
        return results

    def predict(self, img, conf):
        """
        Perform segmentation prediction on an input image.
        
        Args:
            img: Input image (can be path or numpy array)
            conf (float): Confidence threshold for predictions
                         Only predictions above this threshold are returned
        
        Returns:
            results: Model predictions including:
                    - Segmentation masks
                    - Bounding boxes
                    - Confidence scores
                    - Class predictions
        """
        # Run prediction with specified confidence threshold
        results = self.model.predict(img, conf=conf)
        return results