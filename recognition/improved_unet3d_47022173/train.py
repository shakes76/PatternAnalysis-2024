from dataset import *
images_path = "./data/semantic_MRs_anon/"
labels_path = './data/semantic_labels_anon/'
data = ProstateDataset3D(images_path, labels_path, None, None)