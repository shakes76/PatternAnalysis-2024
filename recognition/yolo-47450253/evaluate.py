import torch
from ultralytics import YOLO

DATA_YML = "./data.yml"
TRAINED_WEIGHTS = "./models/train2/weights/best.pt"

'''
evaluate()
    Evaluates the model in TRAINED_WEIGHTS against the test dataset referenced by DATA_YML.
'''
def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #default to gpu
    model = YOLO(TRAINED_WEIGHTS).to(device)

    validate = model.val(data=DATA_YML, split='test')

if __name__ == '__main__':
    evaluate()