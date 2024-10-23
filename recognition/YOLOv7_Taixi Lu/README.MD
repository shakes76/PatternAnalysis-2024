# YOLOv7 for Skin Lesion identification


This is a project using YOLOv7 (You Only Look Once v7) model to identify and classify skin lesions. The dataset of skin lesions was retrieved from ISIC Challenge Datasets 2017(https://challenge.isic-archive.com/data/#2017). Except the training data and test data, the corresponding gold standard lesion diagnoses and binary mask images are used for classification and IoU (Intersection over Union) calculation. 

## Model summary
The YOLO model is a real time visual object detection model. It uses gride to divide the input images, and each grid cell detects the objects inside. For each grid cell, YOLO output groups of boundary boxes and corresponding confidence level, class features. When there are multiple overlapping bounding boxes, YOLO uses a non-maximum suppression method to select the best box and ignore redundant results.

## My implementation
### Data loading
I downloaded training data and test data and the corresponding lesion diagnoses and binary mask images and put them in "/data/train" and "/data/test" respectively under working directory. 

Then I created an ISICDataset (inherited from Dataset of pyTorch) under sataset.py to load the dataset. I created a custom __getitem__  function to load the data in the format I need. That is:

___For each image, a transform was applied to resize it and transform to tensor type.___ 

___For each diagnoses, I manually checked that there are no instances of having melanoma and seborrheic keratosis at the same time. Thus, I re-encoded them for classification. More specifically, 1 for having melanoma, 2 for seborrheic keratosis, and 0 for not having none of them.
	I also created a get_dataloader function to load the ISICDataset using DataLoader of pyTorch.___ 

### Model define
Since the YOLOv7 model is very complex, we are allowed to use pre-trained models. I was not able to find a pre-trained YOLOv7 model specifically for the ISIC 2017 dataset.

Then, I first tried to load the standard YOLOv7 model (yolov7.pt, downloaded from https://github.com/WongKinYiu/yolov7/releases/tag/v0.1y) and use an independent classifier to classify based on the 80 one hot encoded class feature in outputs. But the performance was really limited, I got an average IoU around 17%. 

Thus, I decided to train a YOLOv7 specifically for our dataset. I loaded the structure of the standard YOLOv7(yolov7_training.pt, downloaded from https://github.com/WongKinYiu/yolov7/releases/tag/v0.1y) and adjusted base on that. To better fit our dataset, I adjusted the detection layer of YOLOv7. 
    
Firstly, I wrote a helper function to extract the ground truth boundary box from the ground truth mask images. I printed out the height and width of some samples of the boundary box to construct anchors that suit the dimensions for our data. 
    
Then, I modified the class number by replacing the detection layer. In my instance the original number of classes defined in the pretrained YOLO was 80, and I changed it to 3 to represent the 3 possible diagnoses.

I also implemented a customer loss function which calculates and combined the loss of boundary box matching, confidence level matching, and classification matching. A construct_yolo_target function was implemented to help me transform the mask image into standard YOLO boxes. 

### Training