from modules import *
from dataset import *

df = BalancedMelanomaDataset(
    image_shape=(256, 256),
    batch_size=64,
    validation_split=0.3,
    testing_split=0.3,
    balance_split=0.5
)

model = SiameseNetwork(
    image_shape=(256, 256)
)

model.load_classification_model("checkpoints/best_epoch.keras")

# model.plot_embeddings()
model.evaluate_classifier(df.dataset_val)