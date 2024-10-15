from skimage.metrics import structural_similarity as ssim


def calculate_ssim(original, reconstructed):
    ssim_value = ssim(original, reconstructed, data_range=original.max() - original.min())
    return ssim_value