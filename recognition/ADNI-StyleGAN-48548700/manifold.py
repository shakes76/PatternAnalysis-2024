"""
File: manifold.py
Author: Baibhav Mund (ID: 48548700)

Description:
t-SNE Visualization of Style Codes from AD/NC datsets

This script loads style codes from AD and NC datasets, combines them, and applies t-SNE
to reduce the dimensionality for visualization. The resulting embeddings are plotted 
in 2D, color-coded by dataset to highlight distinctions between the style codes.

Dependencies:
    - numpy
    - scikit-learn (for t-SNE)
    - matplotlib (for visualization)

Usage:
    Ensure 'style_codes_log_AD.npy' and 'style_codes_log_NC.npy' 
    are in the working directory.

Output:
    - A saved plot ("tsne_style_codes_datasets.png") displaying the t-SNE 
      embeddings, with Dataset 1 in blue and Dataset 2 in red.

"""
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load style codes from each dataset
style_codes_log_AD = np.load('./gen_epochs/run_AD/style_codes_log.npy', allow_pickle=True)
style_codes_log_NC = np.load('./gen_epochs/run_NC/style_codes_log.npy', allow_pickle=True)

# Concatenate style codes within each dataset to create single arrays
style_codes_all_AD = np.concatenate(style_codes_log_AD, axis=0)
style_codes_all_NC = np.concatenate(style_codes_log_NC, axis=0)

# Create labels for each dataset (e.g., 0 for Dataset 1 and 1 for Dataset 2)
labels_1 = np.zeros(style_codes_all_AD.shape[0], dtype=int)
labels_2 = np.ones(style_codes_all_NC.shape[0], dtype=int)

# Combine style codes and labels
style_codes_combined = np.concatenate([style_codes_all_AD, style_codes_all_NC], axis=0)
labels_combined = np.concatenate([labels_1, labels_2])

# Apply t-SNE to the combined style codes
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
tsne_embeddings = tsne.fit_transform(style_codes_combined)

# Plot the t-SNE embeddings, color-coded by dataset
plt.figure(figsize=(10, 8))
plt.scatter(
    tsne_embeddings[labels_combined == 0, 0], 
    tsne_embeddings[labels_combined == 0, 1], 
    s=10, color='blue', label='AD', alpha=0.5
)
plt.scatter(
    tsne_embeddings[labels_combined == 1, 0], 
    tsne_embeddings[labels_combined == 1, 1], 
    s=10, color='red', label='CN', alpha=0.5
)

# Customize plot
plt.title("t-SNE Visualization of Style Codes")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.savefig("tsne_style_codes.png")
plt.show()

