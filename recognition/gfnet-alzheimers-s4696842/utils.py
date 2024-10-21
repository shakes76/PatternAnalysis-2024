import torch


def get_device():
    """
    Determines the best available device ('cuda', 'mps', or 'cpu') for computation.

    Returns:
        str: The device identifier ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# AD = Alzheimers Disease. NC = Normal Control
ADNI_CLASSES = ["AD", "NC"]
