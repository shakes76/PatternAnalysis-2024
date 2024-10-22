from modules import yolo_model
import torch

yaml_path = "data/ISIC" # path TO (but before yaml): also want processed test/train/val here

def run_train(yaml_path):
    model = yolo_model("yolov7.pt") # initial weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = model.train(batch=32, device=device, data=f"{yaml_path}/isic.yaml", epochs=200, imgsz=512)

if __name__ == '__main__':
    run_train(yaml_path)