import modules

model = modules.YOLOv11Medium()

# training
results = model.train(data = r'Data/lesion_detection.yaml', epochs = 50, save = True)
