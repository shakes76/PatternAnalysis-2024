import numpy as np
import torch
from modules import yolo_model
from dataset import scan_directory, get_newest_item, iou_torch


modified_filepath = "./datasets/ISIC" # file path for processed dataset

def run_train():
    model = yolo_model("yolov8s.pt") # initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.train(batch=16, device=device, data=f"datasets/isic.yaml", epochs=2, imgsz=512)
    # results = model.train(batch=32, device=device, data=f"{yaml_path}/isic.yaml", epochs=200, imgsz=512) # used
    # results = model.train(batch=16, device=device, data=f"{yaml_path}/isic.yaml", epochs=2, imgsz=512)

def run_test(run_number=-1, partition="test"):
    """
    Parameters:
        trained_model: path to the trained model .pt file
        partition: 'train'/'test'/'val'
    
    to run test, just run inference on test set - get the bounding boxes and compare?
    makes the most sense to me
    """
    if run_number == -1: # most recent training run's best weights
        path = get_newest_item("./runs/detect/") + "/weights/best.pt"
    else:
        path = f"./runs/detect/train{run_number}/weights/best.pt"
    model = yolo_model(path)
    print(f"\bWeights used for test: {path}\n_______________________________________\n")
    
    ious = []
    for file in scan_directory(partition): 
        results = model.predict(f"{modified_filepath}/{partition}/images/ISIC_{file}.jpg", imgsz=512, conf=0.25) 
        for result in results:
            pred_bbox = result.boxes.xywh
            if pred_bbox.size()==torch.Size([0, 4]):
                iou = 0
            else:
                if pred_bbox.size() != torch.Size([1, 4]):
                    pred_bbox = [pred_bbox[0]] # takes the most confident classification if multiple exist

                pred_bbox = pred_bbox 
                true_xywh = [float(i)*512 for i in open(f"{modified_filepath}/{partition}/labels/ISIC_{file}.txt").read().split(" ")[1:]]
                true_xywh=torch.tensor([true_xywh])
                iou = iou_torch(pred_bbox, true_xywh)

            print(f"IoU for sample={iou}")
            ious.append(iou)

    print(f"\naverage IoU={np.array(ious).mean()}")
    # print()
    print(f"{sum(1 for v in ious if v >= 0.8)} out of {len(ious)} samples had IoU>=0.8")



if __name__ == '__main__':
    # run_train()
    run_test()