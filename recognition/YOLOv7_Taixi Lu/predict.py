import torch
from modules import get_yolo_model
from dataset import get_dataloader
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt

def predict_yolo(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            preds = model(images)

            for i, pred in enumerate(preds):
                image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
                image = (image * 255).astype(np.uint8)  # Assuming image is normalized between 0 and 1

                # Draw predictions on the image
                if pred is not None and len(pred) > 0:
                    draw = ImageDraw.Draw(Image.fromarray(image))
                    for p in pred:
                        x1, y1, x2, y2, conf, cls = p[:6]
                        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                        draw.text((x1, y1), f"{cls:.0f}:{conf:.2f}", fill='red')

                # Plot and save the result
                plt.figure()
                plt.imshow(image)
                plt.title(f"Prediction for {img_names[i]}")
                plt.axis('off')
                plt.savefig(f"predictions/{img_names[i]}_prediction.png")
                plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader(csv_file='data/validation/ISIC-2017_Validation_Part3_GroundTruth.csv', root_dir='data/validation',
                                batch_size=1)
    yolov7 = get_yolo_model(model_path='trained model/yolov7_epoch_20.pt', device=device)

    # Run prediction on validation dataset
    predict_yolo(yolov7, dataloader, device=device)
