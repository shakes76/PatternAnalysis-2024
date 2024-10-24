""" Execute this file if you want to test the pre-trained saved model"""
import torch
from dataset import FacebookPagePageLargeNetwork
from modules import GCN
from sklearn.metrics import accuracy_score
from utils import SEED

from utils import plot_tsne

FILE_PATH = "./GCN_model_state_dict.pth"
DATA_FILE_PATH = "./facebook.npz"
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.1

OUT_CHANNELS = 64
# government, tvshow, companies, politicians
NUM_CLASSES = 4


def test_GCN(dataset: FacebookPagePageLargeNetwork, gcn: GCN):

    gcn.eval()
    with torch.no_grad():
        outputs = gcn(dataset)

        # Get the prdictions
        y_test_predicted = outputs[dataset.test_mask]

        _, predicted = torch.max(y_test_predicted, 1)
        predicted = predicted.cpu().numpy()

        test_accuracy = accuracy_score(dataset.y_test.cpu().numpy(), predicted)
        print(f"Test Accuracy: {test_accuracy}")
        plot_tsne(y_test_predicted.cpu().numpy(), dataset.y_test.cpu().numpy())

if __name__ == "__main__":

    device = torch.device( 'gpu' if torch.cuda.is_available() else 'cpu')
    # Create the dataset and transfer it to device
    dataset = FacebookPagePageLargeNetwork(DATA_FILE_PATH, TEST_RATIO, VALIDATION_RATIO, SEED)
    dataset.to(device)

    # Load the saved state dictionary
    state_dict = torch.load(FILE_PATH, weights_only=True)

    gcn = GCN(dataset.features.shape[1], OUT_CHANNELS, NUM_CLASSES)
    gcn.load_state_dict(state_dict)
    print(f"Device: {device}")

    test_GCN(dataset, gcn)
