""" Execute this file if you want to test the pre-trained saved model"""
import torch
from dataset import FacebookPagePageLargeNetwork
from modules import GCN
from sklearn.metrics import accuracy_score
from utils import SEED

FILE_PATH = "./GCN_model.pth"
DATA_FILE_PATH = "./facebook.npz"
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1


def test_GCN(dataset: FacebookPagePageLargeNetwork, gcn: GCN):

    gcn.eval()
    with torch.no_grad():
        outputs = gcn(dataset)

        # Get the loss:
        y_test_predicted = outputs[dataset.test_mask]
        # validation_loss = criterion(y_validation_predicted, dataset.y_test)
        
        _, predicted = torch.max(y_test_predicted, 1)
        predicted = predicted.cpu().numpy()

        test_accuracy = accuracy_score(dataset.y_test.cpu().numpy(), predicted)

        # total_validation_loss.append(validation_loss.item())
        # total_validation_accuracy.append(validation_accuracy)

        
        print(f"Test Accuracy: {test_accuracy}")

if __name__ == "__main__":

    gcn = torch.load(FILE_PATH, weights_only=False)
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Create the dataset and transfer it to device
    dataset = FacebookPagePageLargeNetwork(DATA_FILE_PATH, TEST_RATIO, VALIDATION_RATIO, SEED)
    dataset.to(device)

    test_GCN(dataset, gcn)
