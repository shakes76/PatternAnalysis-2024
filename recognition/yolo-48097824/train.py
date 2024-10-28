import time
import datetime
from modules import assign_device, use_yolo

def initialise():

    device = assign_device()  
    model = use_yolo(device)  
    return device, model

def execute_training(model, device):

    return model.train(
        batch = 4,  
        device = device,
        model = model,
        data = "/home/jared-cc/PatternAnalysis-2024/recognition/yolo-48097824/traindata.yaml",
        epochs = 80,
        imgsz = 640
    )

def timer(start_time):

    end_time = time.time()
    print("End Time:", time.ctime(end_time))
    print("Time Taken to Train (H:M:S):", str(datetime.timedelta(seconds=end_time - start_time)))

def main():

    start_time = time.time()  
    print("Start time:", time.ctime(start_time))
    
    device, model = initialise() 

    try:
        results = execute_training(model, device)
        print("Training completed successfully.")

    except Exception as e:
        print("Error during training:", str(e))

    timer(start_time)  

if __name__ == "__main__":
    main()
