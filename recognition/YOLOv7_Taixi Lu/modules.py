import os
import sys

sys.path.append('./yolov7')
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as func
from models.yolo import IDetect

"""
YOLOv7 installed (in the working directory) using cmd:
git clone https://github.com/WongKinYiu/yolov7.git
pip install -r requirements.txt
"""

g_num_classes = 3
g_num_anchors = 3


def get_yolo_model(model_path, device):
    """
    Using a pre-trained YOLOv7 models found in
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1
    use cpu unless specified to enhance stability across platforms
    """
    model = attempt_load(model_path, map_location=device)
    model.to(device)
    # model.requires_grad_(True)
'''
    # Change the detection layer to output 3 classes only
    for layer in model.model.modules():
        if isinstance(layer, IDetect):
            for i, conv_layer in enumerate(layer.m):
                new_out_channels = g_num_anchors * (g_num_classes + 5)
                layer.m[i] = torch.nn.Conv2d(conv_layer.in_channels, new_out_channels,
                                             kernel_size=conv_layer.kernel_size,
                                             stride=conv_layer.stride,
                                             padding=conv_layer.padding)
'''

    # Change the detection layer to output 3 classes only
    for name, module in model.model._modules.items():
        if isinstance(module, IDetect):
            # print(f"Original IDetect: \n{module}")
            # print(f"\nOriginal anchors: \n{module.anchors}")
            new_anchors = module.anchors.view(3, 6)  # flatten to suit the yolo config file (yaml) format
            new_idetect = IDetect(
                nc=g_num_classes, anchors=new_anchors, ch=[layer.in_channels for layer in module.m])
            new_idetect.f = module.f
            new_idetect.i = module.i
            model.model._modules[name]=new_idetect
            print(f"\nModified IDetect: \n{model.model._modules[name]}")
            # print(f"\nModified anchors: \n{model.model._modules[name].anchors}")
    # print(f"Model loaded: \n{model}")
    model.to(device)
    return model


def calculate_iou(yolo_output, ground_truth):
    # Calculate Intersection Over Union (IoU) between YOLO output and ground truth mask
    ious = []
    for yolo_box in yolo_output:
        yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box[:4]
        gt_indices = np.argwhere(ground_truth > 0)
        if len(gt_indices) == 0:
            continue
        gt_x1, gt_y1 = gt_indices.min(axis=0)
        gt_x2, gt_y2 = gt_indices.max(axis=0)

        inter_x1 = max(yolo_x1, gt_x1)
        inter_y1 = max(yolo_y1, gt_y1)
        inter_x2 = min(yolo_x2, gt_x2)
        inter_y2 = min(yolo_y2, gt_y2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        union_area = yolo_area + gt_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        ious.append(iou)

    return np.mean(ious) if len(ious) > 0 else 0


def get_YOLO_box(ground_truth):
    gt_indices = np.argwhere(ground_truth > 0)
    if len(gt_indices) == 0:
        return None
    gt_x1, gt_y1 = gt_indices.min(axis=0)
    gt_x2, gt_y2 = gt_indices.max(axis=0)

    x_center = (gt_x1 + gt_x2) / 2
    y_center = (gt_y1 + gt_y2) / 2
    width = gt_x2 - gt_x1
    height = gt_y2 - gt_y1

    return x_center, y_center, width, height


def construct_yolo_target(ground_truth, grid_size, target_class, image_size=(640, 640), num_classes=3, num_anchors=3):
    # Create an empty target tensor
    target = torch.zeros((num_anchors, grid_size, grid_size, 5 + num_classes))

    # Extract bounding box information
    yolo_box = get_YOLO_box(ground_truth)
    if yolo_box is None:
        return target

    x_center, y_center, width, height = yolo_box

    # Normalize the box coordinates
    img_height, img_width = image_size
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    # Find the corresponding grid cell
    grid_x = int(x_center * grid_size)
    grid_y = int(y_center * grid_size)

    # Calculate the cell-relative position
    x_cell = (x_center * grid_size) - grid_x
    y_cell = (y_center * grid_size) - grid_y

    # Fill in the target tensor
    for anchor in range(num_anchors):
        target[anchor, grid_y, grid_x, 0] = x_cell  # tx
        target[anchor, grid_y, grid_x, 1] = y_cell  # ty
        target[anchor, grid_y, grid_x, 2] = width  # tw
        target[anchor, grid_y, grid_x, 3] = height  # th
        target[anchor, grid_y, grid_x, 4] = 1  # Objectness score
        target[0, grid_y, grid_x, 5 + target_class] = 1  # One-hot encoded class

    return target


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord  # Weight for bounding box localization loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence loss

    def forward(self, predictions, targets):
        """
        predictions: list of tensors, each with shape [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
                     Contains [tx, ty, tw, th, object_confidence, class_probs...]
        targets: list of tensors, similar structure to predictions but with ground-truth labels.
        """
        total_loss = 0

        # Iterate over each scale prediction
        for pred, target in zip(predictions, targets):
            # Separate predictions and targets into different components
            pred_boxes = pred[..., :4]  # tx, ty, tw, th
            pred_obj = pred[..., 4]  # object confidence
            pred_class = pred[..., 5:]  # class probabilities

            target_boxes = target[..., :4]  # Ground-truth boxes
            target_obj = target[..., 4]  # Ground-truth objectness
            target_class = target[..., 5:]  # Ground-truth classes
            target_class = target_class[:, 0, :, :, :]

            # Bounding box loss (using Mean Squared Error)
            box_loss = func.mse_loss(pred_boxes, target_boxes, reduction='sum')

            # Objectness loss (using Binary Cross-Entropy)
            obj_loss = func.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='sum')

            # No-object loss
            noobj_loss = func.binary_cross_entropy_with_logits(pred_obj, target_obj,
                                                               reduction='sum') * self.lambda_noobj

            # Classification loss (using Cross-Entropy)
            class_loss = func.cross_entropy(pred_class, target_class.long(), reduction='sum')

            # Combine the losses
            total_loss += (self.lambda_coord * box_loss) + obj_loss + noobj_loss + class_loss

        return total_loss
