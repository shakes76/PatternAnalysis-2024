import sys
sys.path.append('./yolov7')
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.general import non_max_suppression

"""
YOLOv7 installed (in the working directory) using cmd:
git clone https://github.com/WongKinYiu/yolov7.git
pip install -r requirements.txt
"""

class YOLOv7Model:
    def __init__(self, model_path='yolov7.pt', device='cpu'):
        """
        Using a pre-trained YOLOv7 models found in
        https://github.com/WongKinYiu/yolov7/releases/download/v0.1
        use cpu unless specified to enhance stability across platforms
        """
        self.device = torch.device(device)
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

    def forward(self, images, conf_thres=0.2, iou_thres=0.45):
        with torch.no_grad():
            preds = self.model(images)
            results = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=iou_thres)
        return results


class ClassificationYOLO(nn.Module):
    """
    An independent classifier for convert the output features of YOLO to labels
    """
    def __init__(self, input_dim, num_classes=2):
        super(ClassificationYOLO, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)
