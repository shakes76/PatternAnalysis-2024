import nibabel as nib
import matplotlib.pyplot as plt

path = r'C:\Users\GamingPC\Desktop\COMP3710-Project\recognition\VQVAE_s4803279\HipMRI_study_keras_slices_data\keras_slices_train\case_004_week_0_slice_0.nii.gz'

Nifti_img  = nib.load(path)
print(Nifti_img.shape)

nii_data = Nifti_img.get_fdata()
nii_aff  = Nifti_img.affine
nii_hdr  = Nifti_img.header
# print(nii_aff ,'\n',nii_hdr)
print(nii_data.shape)

# For 3D images (e.g., 256 x 256 x N slices)
if len(nii_data.shape) == 3:
    for slice_number in range(nii_data.shape[2]):
        plt.imshow(nii_data[:, :, slice_number], cmap = 'gray')
        plt.title(f'Slice {slice_number}')
        plt.show()

# For 2D images (e.g., a single slice)
elif len(nii_data.shape) == 2:
    plt.imshow(nii_data, cmap='gray')
    plt.title('2D Image')
    plt.show()