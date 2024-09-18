from torchvision import transforms
from dataset import get_dataloader

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader_AD = get_dataloader('data/train/AD', batch_size=32, transform=image_transform)
train_loader_CN = get_dataloader('data/train/CN', batch_size=32, transform=image_transform)