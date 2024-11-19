"""This is the file when the training is happening.
    Data is loaded, The model is created, saved and the results are
    plotted and save here
"""
from modules import GCN
from dataset import FacebookPagePageLargeNetwork
from sklearn.metrics import accuracy_score
from utils import save_plot, SEED
import torch


FILE_PATH = "./facebook.npz"
VALIDATION_RATIO = 0.1
TEST_RATIO= 0.1

EPOCHS = 400
LEARING_RATE = 0.01

OUT_CHANNELS = 64
# government, tvshow, companies, politicians
NUM_CLASSES = 4


def train_GCN(dataset: FacebookPagePageLargeNetwork, gcn: GCN):
    # Define optimizer and Loss criterion
    optimizer = torch.optim.Adam(gcn.parameters(), lr=LEARING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    total_train_accuracy = []
    total_train_loss = []

    total_validation_accuracy = []
    total_validation_loss = []

    # Arbitrary
    best_accuracy = 0
    best_state_dict = None

    for i in range(EPOCHS):
        gcn.train()

        # Have to pass the whole dataset since spliting the adjacency matrix into train/test/validation can be problematic
        outputs = gcn(dataset)

        # Get the loss:
        y_train_predicted = outputs[dataset.train_mask]
        loss = criterion(y_train_predicted, dataset.y_train)
        
        _, predicted = torch.max(y_train_predicted, 1)
        predicted = predicted.cpu().numpy()

        accuracy = accuracy_score(dataset.y_train.cpu().numpy(), predicted)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        total_train_loss.append(loss.item())
        total_train_accuracy.append(accuracy)

        #  Do the evaluation
        gcn.eval()
        outputs = gcn(dataset)

        # Get the loss:
        y_validation_predicted = outputs[dataset.validation_mask]
        validation_loss = criterion(y_validation_predicted, dataset.y_val)
        
        _, predicted = torch.max(y_validation_predicted, 1)
        predicted = predicted.cpu().numpy()

        validation_accuracy = accuracy_score(dataset.y_val.cpu().numpy(), predicted)

        total_validation_loss.append(validation_loss.item())
        total_validation_accuracy.append(validation_accuracy)

        # Save the best parameters of the model
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_state_dict = gcn.state_dict()

        # Print the validation and train loss every 50 epochs:
        if i % 10 == 0 or i == EPOCHS - 1:
            print(f"Epoch: {i + 1}/{EPOCHS}\nTRAIN: Loss: {loss.item()}, Accuracy: {accuracy}\nVALIDATION: Loss: {validation_loss.item()}, Accuracy: {validation_accuracy}\n")

    # Plot and save the loss and accuracy
    save_plot(total_train_accuracy, total_validation_accuracy, EPOCHS, "./images/accuracy.png", "Accuracy", (0, 1))
    save_plot(total_train_loss, total_validation_loss, EPOCHS, "./images/loss.png", "Loss", (0.5, 2))
    

    # Save the state dictionary instead of the whole model
    torch.save(best_state_dict, "./GCN_model_state_dict.pth")


if __name__ == "__main__":

   
    device = torch.device( 'gpu' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create the dataset and transfer it to device
    dataset = FacebookPagePageLargeNetwork(FILE_PATH, TEST_RATIO, VALIDATION_RATIO, SEED)
    dataset.to(device)

    # Create the model
    gcn = GCN(dataset.features.shape[1], OUT_CHANNELS, NUM_CLASSES)
    train_GCN(dataset, gcn)
