import torch

def getAccuracy(test_dataloader, model, device, max_subset : int = -1):
    with torch.no_grad():
        total_correct = 0
        total_images = 0
        for batch, (images, targets) in enumerate(test_dataloader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            most_likely = torch.max(outputs, dim=1).indices #get class with highest score
            correct = most_likely == targets
            total_correct += correct.sum()
            total_images += len(images)

            #print(f"[{batch}/{len(test_dataloader)}] Batch accuracy: {correct.sum()/len(images)}")
            if max_subset != -1 and batch > max_subset:
                break
        return total_correct/total_images
