import torch
from ultralytics import YOLO

TRAINED_WEIGHTS = "./models/train2/weights/best.pt"
TEST_SET = "./data/images/test"

'''
predict()
    Performs predictions on the test image set (TEST_SET) using a given trained model (TRAINED_WEIGHTS). Saves images to runs/detect/predictN
'''
def predict():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
    model = YOLO(TRAINED_WEIGHTS).to(device)

    results = model.predict(source=TEST_SET,
                            imgsz=640,
                            conf=0.5,
                            iou=0.8,
                            save=False)


if __name__ == '__main__':
    predict()