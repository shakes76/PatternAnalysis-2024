from ultralytics import YOLO

class YOLOSegmentation:
    def __init__(self, weights_path):
        self.model = YOLO(weights_path)

    def train(self, params):
        results = self.model.train(**params)
        return results

    def evaluate(self):
        results = self.model.val()
        return results

    def predict(self, img, conf):
        results = self.model.predict(img, conf=conf)
        return results


