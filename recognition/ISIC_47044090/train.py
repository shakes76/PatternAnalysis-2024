from modules import yolo_model
import torch
import os

yaml_path = "./datasets" # path TO (but before yaml): also want processed test/train/val here
# trained_model = ""

def run_train(yaml_path):
    model = yolo_model("yolov7.pt") # initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.train(batch=32, device=device, data=f"{yaml_path}/isic.yaml", epochs=200, imgsz=512)

def run_test(run_number=-1):
    """
    Parameters:
        trained_model: path to the trained model .pt file
    
    to run test, just run inference on test set - get the bounding boxes and compare?
    makes the most sense to me
    """
    if run_number == -1:
        path = list(os.scandir(f"./runs/detect"))[-1].path + "/weights/best.pt"
    else:
        path = f"./runs/detect/train{run_number}/weights/best.pt"
    print(f"Weights used for test: {path}")
    
    model = yolo_model(path)
    results = model(["./datasets/ISIC/test/images/ISIC_0001769.jpg"]) 
    for result in results:
        print(result.boxes)

if __name__ == '__main__':
    # run_train(yaml_path)
    run_test()