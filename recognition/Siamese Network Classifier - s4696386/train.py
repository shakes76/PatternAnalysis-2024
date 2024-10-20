import torch, numpy

from modules import SiameseNetwork
from dataset import APP_MATCHER, read_data, DEFAULT_LOCATION, PROCESSED_DATA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEFAULT_SAVE_LOCATION = "E:/COMP3710 Project/PatternAnalysis-2024/"

# With support from:
# https://github.com/pytorch/examples/blob/main/siamese_network

def train(model: SiameseNetwork, device, train_loader, optimizer, epoch, log_interval, dry_run = False, verbose = False):
    model.train()
    
    criterion = model.loss_criterion

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs: torch.Tensor = model(images_1, images_2).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if verbose:
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx*len(images_1)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader)}%)]")
                print(f"Loss: {loss.item()}")
        if dry_run:
            break

def test(model: SiameseNetwork, device, test_loader, epoch: int, threshold = 0.5, verbose = False):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    benign_count = 0

    criterion = model.loss_criterion

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss

            # Store true and predicted values
            all_targets.extend(targets.cpu().numpy())
            pred = torch.where(outputs > threshold, 1, 0)  # get the index of the max log-probability
            all_preds.extend(pred.cpu().numpy())
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    benign_count = all_preds.count(0)

    if verbose:
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')
        # print(f"Predictions: {all_preds}")
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        print(f"Predicted {benign_count} benigns")
        print(f"Epoch: {epoch}, Threshold: {threshold}\n")
    

    return accuracy

def run_model(batch_size: int, epochs: int, learning_rate: int, processed_data: PROCESSED_DATA, threshold: float = 0.5):
    print("="*200)
    print(f"Training with: batch_size {batch_size}, epochs {epochs}, learning_rate {learning_rate}")
    torch_seed = 10
    shuffle = True
    gamma = 0.7
    log_interval = 10
    dry_run = False
    

    torch.manual_seed(torch_seed)

    # Use cuda device if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    

    # Create Datasets and place them into DataLoaders
    train_dataset = APP_MATCHER(processed_data, train=True)
    test_dataset = APP_MATCHER(processed_data, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle)

    # Create the model
    model = SiameseNetwork().to(device)
    # Define which optimizer to use
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    
    # run epochs
    for epoch in range(1, epochs+1):
        test_verbose = epoch == epochs
        train(model, device, train_loader, optimizer, epoch, log_interval, dry_run, verbose=False)
        accuracy = test(model, device, test_loader, epoch, verbose=test_verbose, threshold=threshold)
        save_model = accuracy >= 0.95
        if save_model:
            save_loc = f"{DEFAULT_SAVE_LOCATION}siamese_network_{batch_size}_({epoch}.{epochs})_{learning_rate}_{threshold}.pt"
            torch.save((model.state_dict(), optimizer.state_dict()), save_loc)
            print(f"Saved to {save_loc} with {accuracy}")
        scheduler.step()


    return accuracy, batch_size, epochs, learning_rate, threshold
        

def main():

    processed_data: PROCESSED_DATA = read_data(DEFAULT_LOCATION[0], DEFAULT_LOCATION[1])

    batch_sizes = [16]
    epochs = [40]
    learning_rates = [i/10000 for i in range(2, 6)]
    thresholds = [0.3, 0.4, 0.6, 0.7]

    accuracies = []

    os.chdir(DEFAULT_SAVE_LOCATION)
    print(os.getcwd())

    for th in thresholds:
        for bs in batch_sizes:
            for e in epochs:
                for lr in learning_rates:
                    accuracy = run_model(bs, e, lr, processed_data, threshold=th)
                    accuracies.append(accuracy)
                    with open("results.txt", "a") as results:
                        results.write(str(accuracy) + "\n")

    print(max(accuracies, key=lambda x: x[0]))

# Only run main when running file directly (not during imports)
if __name__ == "__main__":
    import cProfile, pstats, os

    current_directory = os.getcwd()
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    os.chdir(current_directory)
    stats.dump_stats(filename="profile.prof")