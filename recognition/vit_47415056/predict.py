import torch
from modules import initialize_model
from dataset import get_test_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_and_visualize(model_path, test_data_dir):
    pass

if __name__ == "__main__":
    model_path = "model_weights.pth"
    test_data_dir = "/home/groups/comp3710/ADNI/AD_NC/test"
    predict_and_visualize(model_path, test_data_dir)