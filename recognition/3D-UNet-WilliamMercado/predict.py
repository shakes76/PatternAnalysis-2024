import matplotlib.pyplot as plt
import torch
from const import DATASET_PATH, NET_OUTPUT_TARGET
from modules import FullUNet3D
from dataset import load_data_3d

# CHECK CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("NO CUDA AVAILABLE. CPU IN USE")
device = 'cpu' # Force Cpu usage for low power computers
print(device)

TARGET_LAYERS = [32, 64, 96]
TEST_FILE = "/semantic_MRs/B006_Week0_LFOV.nii.gz"
LABEL_FILE = "/semantic_labels_only/B006_Week0_SEMANTIC.nii.gz"

input_img = torch.Tensor(load_data_3d([DATASET_PATH + TEST_FILE])).unsqueeze(0)
expected_out = torch.Tensor(load_data_3d([DATASET_PATH + LABEL_FILE])).unsqueeze(0)
print(input_img.shape)
model:FullUNet3D = torch.load(NET_OUTPUT_TARGET,weights_only=False)
model = model.to(device=device)

model.eval()
with torch.no_grad():
    out:torch.Tensor = model(input_img)
    print(out.shape)
    _, out = out.max( dim=1, keepdim= True)

print(out.shape)

f, axarr = plt.subplots(len(TARGET_LAYERS),3)
for col, layer in enumerate(TARGET_LAYERS):
    for row, to_show in enumerate([input_img, out, expected_out]):
        axarr[row,col].imshow(to_show[0,0,:,:,layer])

plt.show()
