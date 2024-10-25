from ultralytics import YOLO
import random
import cv2
import numpy as np

model = YOLO("~/weights/best.pt")
img = cv2.imread("")
conf = 0.4
results = model.predict(img, conf=conf)
color = random.choices(range(256), k=3)

for result in results:
    for mask, box in zip(result.masks.segments, result.boxes):
        points = np.int32([np.float64(mask) * box.xywh.numpy()[0][2]])
        cv2.fillPoly(img, points, color)

cv2.imwrite("prediction_test.jpg", img)
