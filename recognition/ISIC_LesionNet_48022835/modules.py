"""
Can be called explicitly to install commonly used weights for different model sizes.
Imported into other scripts to access a YOLO model w/ given weights.

Usage example: python modules.py | to install weights for a small, medium and large model.

@author Ewan Trafford
"""

from ultralytics import YOLO

def YOLOv11(weights_path):
    """
    Loads an instance of a YOLOv11 model with weights specified.

    Args:
        weights_path (string): The absolute or relative path to the .pt file containing model weights.

    Returns:
        YOLO(weights_path): A YOLOv11 model with the given weights installed.
    """
    
    return YOLO(weights_path)

# Run modules.py to pre-install weights for useful models | not explicitly required
if __name__ == "__main__":
    model_s = YOLOv11("yolo/yolo11s.pt")
    model_m = YOLOv11("yolo/yolo11m.pt")
    model_l = YOLOv11("yolo/yolo11l.pt")