import numpy as np
import torch
import os
from modules import UNet3D
import nibabel as nib
import torchvision.transforms as transforms
from PIL import Image

os.chdir(os.path.dirname(__file__))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet3D().to(device)
model.load_state_dict(torch.load('ImprovedUNET3DsegmentationModel.pth', map_location=device))
model.eval()

image_path = '~/HipMRI_study_complete_release_v1/semantic_MRs_anon/Case_004_Week0_LFOV.nii.gz'
image = nib.load(image_path)
image = np.asarray(image.dataobj)

totensor = transforms.ToTensor()
image = totensor(image)
image = image.unsqueeze(0)
image = image.unsqueeze(0)
image = image.float().to(device)

pred = model(image)
pred = pred.argmax(1)
pred = pred.squeeze(0)
pred = torch.permute(pred, (1,2,0))
image = image.squeeze()
image = torch.permute(image, (1,2,0))

# Move tensors to CPU for processing
image = image.cpu().numpy()
pred = pred.cpu().numpy()

# Normalizing to 0-255 range
image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
pred = (255 * (pred - pred.min()) / (pred.max() - pred.min())).astype(np.uint8)

# Save individual slices
output_dir = 'output_slices'
os.makedirs(output_dir, exist_ok=True)

for i in range(image.shape[2]):  # iterate over the axial slices
    img_slice = Image.fromarray(image[:, :, i])
    pred_slice = Image.fromarray(pred[:, :, i])
    
    img_slice.save(os.path.join(output_dir, f'image_slice_{i}.png'))
    pred_slice.save(os.path.join(output_dir, f'pred_slice_{i}.png'))

print("Slices saved to 'output_slices' directory.")
