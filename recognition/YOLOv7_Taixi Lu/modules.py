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
    https://github.com/WongKinYiu/yolov7/releases/tag/v0.1
    use cpu unless specified to enhance stability across platforms
    """
    model = attempt_load(model_path, map_location=device)
    model.to(device)
    # model.requires_grad_(True)

    # Change the detection layer to output 3 classes only
    for name, module in model.model._modules.items():
        if isinstance(module, IDetect):
            print(f"Original IDetect: \n{module}")
            print(f"\nOriginal anchors: \n{module.anchors}")
            # new_anchors = module.anchors
            new_anchors = torch.tensor([
                [136.0, 100.0], [125.0, 107.0], [170.0, 117.0],  # Small Anchors
                [222.0, 157.0], [226.0, 170.0], [314.0, 237.0],  # Medium Anchors
                [466.0, 430.0], [571.0, 500.0], [630.0, 630.0]  # Large Anchors
            ]).to(device)
            new_anchors = new_anchors.view(3, 6)  # flatten to suit the yolo config file (yaml) format
            new_idetect = IDetect(
                nc=g_num_classes, anchors=new_anchors, ch=[layer.in_channels for layer in module.m])
            new_idetect.f = module.f
            new_idetect.i = module.i
            new_idetect.stride = module.stride
            model.model._modules[name] = new_idetect
            print(f"\nModified IDetect: \n{model.model._modules[name]}")
            print(f"\nModified anchors: \n{model.model._modules[name].anchors}")
    # print(f"Model loaded: \n{model}")
    model.to(device)
    return model


def get_anchor(model):
    """
    get the anchor in current model, only for construct_yolo_target use, not detached
    """
    for name, module in model.model._modules.items():
        if isinstance(module, IDetect):
            # print(f"Original IDetect: \n{module}")
            # print(f"\nOriginal anchors: \n{module.anchors}")
            return module.anchors


def calculate_iou(yolo_box_vertex, ground_truth_vertex, print_log=False):
    # Calculate Intersection Over Union (IoU) between YOLO output and ground truth mask
    gt_x1, gt_y1, gt_x2, gt_y2 = ground_truth_vertex[:4]
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_box_vertex[:4]
    yolo_area = (yolo_x2 - yolo_x1) * (yolo_y2 - yolo_y1)

    inter_x1 = max(yolo_x1, gt_x1)
    inter_y1 = max(yolo_y1, gt_y1)
    inter_x2 = min(yolo_x2, gt_x2)
    inter_y2 = min(yolo_y2, gt_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = yolo_area + gt_area - inter_area
    if union_area > 0:
        if print_log:
            if inter_area > 0:
                print(f"inter_area: {inter_area:.1f}, iou: {inter_area / union_area:.4f}")
            else:
                print(
                    f"gt_w: {(gt_x2 - gt_x1):.1f}, gt_h: {(gt_y2 - gt_y1):.1f}, gt_area: {gt_area:.2f}, ratio: {yolo_area / gt_area:.3f}")

        return inter_area / union_area
    else:
        return 0


def calculate_iou_YOLO_box(yolo_box, ground_truth_box):
    # Calculate Intersection Over Union (IoU) between YOLO output and ground truth mask
    gt_x_center, gt_y_center, gt_width, gt_height = ground_truth_box[:4]
    gt_x1 = gt_x_center - gt_width / 2
    gt_x2 = gt_x_center + gt_width / 2
    gt_y1 = gt_y_center - gt_height / 2
    gt_y2 = gt_y_center + gt_height / 2
    gt_area = gt_width * gt_height

    yolo_x_center, yolo_y_center, yolo_width, yolo_height = yolo_box[:4]
    yolo_x1 = yolo_x_center - yolo_width / 2
    yolo_x2 = yolo_x_center + yolo_width / 2
    yolo_y1 = yolo_y_center - yolo_height / 2
    yolo_y2 = yolo_y_center + yolo_height / 2
    yolo_area = yolo_width * yolo_height

    inter_x1 = max(yolo_x1, gt_x1)
    inter_y1 = max(yolo_y1, gt_y1)
    inter_x2 = min(yolo_x2, gt_x2)
    inter_y2 = min(yolo_y2, gt_y2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = yolo_area + gt_area - inter_area
    if union_area > 0:
        return inter_area / union_area
    else:
        return 0


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


def construct_yolo_target(ground_truth, grid_size, target_class, anchors,
                          image_size=(640, 640), num_classes=3, num_anchors=3):
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

    # Ensure grid cell indices are within bounds
    # grid_x = min(max(0, grid_x), grid_size - 1)
    # grid_y = min(max(0, grid_y), grid_size - 1)

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

    # best_anchor = 0
    # best_iou = 0
    # for i, anchor in enumerate(anchors):
    #     anchor_width, anchor_height = anchor[0]
    #     anchor_width /= img_width
    #     anchor_height /= img_height
    #
    #     # Calculate IoU between the ground-truth box and the anchor box
    #     inter_width = min(width, anchor_width)
    #     inter_height = min(height, anchor_height)
    #     inter_area = inter_width * inter_height
    #     union_area = (width * height) + (anchor_width * anchor_height) - inter_area
    #     iou = inter_area / union_area
    #
    #     if iou > best_iou:
    #         best_iou = iou
    #         best_anchor = i
    #
    # # Fill in the target tensor for the best matching anchor
    # target[best_anchor, grid_y, grid_x, 0] = x_cell  # tx
    # target[best_anchor, grid_y, grid_x, 1] = y_cell  # ty
    # target[best_anchor, grid_y, grid_x, 2] = width  # tw
    # target[best_anchor, grid_y, grid_x, 3] = height  # th
    # target[best_anchor, grid_y, grid_x, 4] = 1  # Objectness score
    # target[best_anchor, grid_y, grid_x, 5 + target_class] = 1  # One-hot encoded class

    return target


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=50, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord  # Weight for bounding box localization loss
        self.lambda_noobj = lambda_noobj  # Weight for no-object confidence loss

    def forward(self, predictions, targets):
        """
        predictions: list of tensors, each with shape [batch_size, num_anchors, grid_size, grid_size, 5 + num_classes]
                     5 + num_classes Contains [tx, ty, tw, th, object_confidence, class_probs...]
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
            # target_class = target_class[:, 0, :, :, :]

            # Bounding box loss (using Mean Squared Error)
            box_loss = func.mse_loss(pred_boxes, target_boxes, reduction='sum')
            # box_loss = calculate_iou_YOLO_box(pred_boxes, target_boxes)

            # No-object loss
            # noobj_loss = func.binary_cross_entropy_with_logits(pred_obj, target_obj,
            #                                                    reduction='sum')
            obj_mask = target_obj > 0  # Mask for object presence
            noobj_mask = target_obj == 0  # Mask for no object presence

            obj_loss = func.binary_cross_entropy_with_logits(pred_obj[obj_mask], target_obj[obj_mask], reduction='sum')
            noobj_loss = func.binary_cross_entropy_with_logits(pred_obj[noobj_mask], target_obj[noobj_mask],
                                                               reduction='sum')

            # Classification loss (using Cross-Entropy)
            # class_loss = func.cross_entropy(pred_class, target_class.float(), reduction='sum')
            class_loss = func.cross_entropy(pred_class[obj_mask], target_class[obj_mask].float(), reduction='sum')

            # Combine the losses
            total_loss += (self.lambda_coord * box_loss) + obj_loss + (self.lambda_noobj * noobj_loss) + class_loss
            # total_loss += (self.lambda_coord * box_loss) + (self.lambda_noobj * noobj_loss)

        return total_loss
