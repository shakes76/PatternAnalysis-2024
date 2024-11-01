import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("cuda")
if not torch.cuda.is_available():
    print("cpu")  
    
class YOLO(nn.Module):

    #REFERENCE: yolov3-tiny.cfg from https://github.com/pjreddie/darknet/blob/master/cfg
    #Used as basis for what layers were needed 
    def __init__(self, num_classes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        layers = []
        filters = [16,32,64,128,256,512]
        in_channels = 3
        #Convulution layers and maxpooling
        for i in filters:
            layers.append(nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1, bias=False))
            in_channels = i
            layers.append(nn.BatchNorm2d(i))
            layers.append(nn.LeakyReLU(0.1, True)) #might be false
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2)) #Hopefully works
        layers.append(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.LeakyReLU(0.1, True))

        layers.append(nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=1, bias=True))
        self.conv_start = nn.Sequential(*layers)

        #Detection layer - given anchors
        self.anchor1 =  [(81,82), (135,169), (344,319)] #Anchors depends on image?
        
        #Route layer could go here
        self.conv_mid = nn.Sequential(
            nn.Conv2d(255, 128, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Upsample(scale_factor=2, mode="bilinear"))
        #Another route layer maybe
        self.conv_end = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=1, bias=True))
        
        #Another detection layer
        self.anchor2 = [(10,14), (23,27), (37,58)]

    def forward(self, x):
        out = self.conv_start(x)
        out = out.data
        a = self.predict_transform(out, 416, self.anchor1, self.num_classes)
        out = self.conv_mid(out)
        out = self.conv_end(out)
        out = out.data
        b = self.predict_transform(out, 416, self.anchor2, self.num_classes)
        return torch.cat((a, b), 1)

    def predict_transform(self, prediction, inp_dim, anchors, num_classes):
        """
        Decodes the output from the convolution layers and arranges the information into a usable format. 
        The below reference was used for a base for this function.
        REFERENCE: refer to reference 2 in README.
        """
        batch_size = prediction.size(0)
        stride =  inp_dim // prediction.size(2)
        grid_size = inp_dim // stride
        bbox_attrs = 5 + num_classes
        num_anchors = len(anchors)
        
        #Rearranges the feature map to (batch_size, number of boxes, box_attributes)
        prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
        prediction = prediction.transpose(1,2).contiguous()
        prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
        #Get the centre_X, centre_Y and object confidence between 1 and 0
        prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
        prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
        #Add the center offsets 
        grid = np.arange(grid_size)
        a,b = np.meshgrid(grid, grid)

        x_offset = torch.FloatTensor(a).view(-1,1)
        y_offset = torch.FloatTensor(b).view(-1,1)

        x_offset = x_offset.to(device)
        y_offset = y_offset.to(device)

        x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

        prediction[:,:,:2] += x_y_offset
        #log space transform height and the width 
        #so that all boxes are on the same scale
        anchors = torch.FloatTensor(anchors)
        anchors = anchors.to(device)

        #arrange the  probabilities of the classes
        anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
        prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
        prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))
        prediction[:,:,:4] *= stride
        return prediction


def calculate_iou(pred, label):
    """
    Caculates the IoUs of a given list of boxes.
    Used to determine accuracy of given bounding boxes.
    Also is a key part of the loss function.
    """
    px, py, pw, ph = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
    lx, ly, lw, lh = label[0], label[1], label[2], label[3]
    box_a = [px-(pw/2), py-(ph/2), px+(pw/2), py+(ph/2)]
    box_b = [lx-(lw/2), ly-(lh/2), lx+(lw/2), ly+(lh/2)]

    # determine the (x, y) of the corners of intersection area
    ax = torch.clamp(box_a[0], min=box_b[0])
    ay = torch.clamp(box_a[1], min=box_b[1])
    bx = torch.clamp(box_a[2], max=box_b[2]) 
    by = torch.clamp(box_a[3], max=box_b[3]) 

    # compute the area of intersection
    intersect = torch.abs(torch.clamp((bx - ax), min=0) * torch.clamp((by - ay), min=0))

    # compute the area of both the prediction and ground-truth
    area_a = torch.abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = torch.abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

    # compute the iou
    iou = intersect / (area_a + area_b - intersect)
    iou = torch.reshape(iou, (776, 3))
    return iou

class YOLO_loss(nn.Module):
    """
    Given one batch at a time, the loss of the predictions is calculated.
    The formulas used to calculate loss are from the reference below.
    REFERENCE: refer to reference 3 in README.
    """
    def __init__(self):
        super(YOLO_loss, self).__init__()

    def forward(pred, label):
        #Constants
        no_object = 0.5 #Puts less emphasis on loss from boxes with no object
        #Rearrange predictions to have one box shape on each line
        boxes = torch.reshape(pred, (776, 3))

        #IoU
        iou = calculate_iou(pred, label)
        iou, best_boxes = torch.max(iou, dim=1)

        #Loss set up
        class_loss = torch.zeros(776)
        coord_loss = torch.zeros(776)
        conf_loss = torch.zeros(776)
        
        #Calculate loss
        i = 0
        for idx in best_boxes:
            box = boxes[i][idx]
            #coordinate loss
            xy_loss = (label[0]-box[0])**2 + (label[1]-box[1])**2
            wh_loss = ((label[0])**(1/2)-(box[0])**(1/2))**2 + ((label[1])**(1/2)-(box[1])**(1/2))**2
            coord_loss[i] = (xy_loss + wh_loss)
            #Check if there was a detection
            if box[4] > 0.8: #There was
                #classification loss
                class_loss[i] = (label[5] - box[5])**2 + (label[6] - box[6])**2
                #confidence loss
                conf_loss[i] = (label[4] - box[4])**2
            else: #There wasn't
                conf_loss[i] = no_object*((label[4] - box[4])**2)
            i += 1
        
        #Final count
        total_loss = 0
        total_loss += torch.sum(coord_loss) 
        total_loss += torch.sum(class_loss)
        total_loss += torch.sum(conf_loss)

        return total_loss

def single_iou(pred, label):
        """
        Calculates the IoU of a single box
        """
        px, py, pw, ph = pred[:,0], pred[:,1], pred[:,2], pred[:,3]
        lx, ly, lw, lh = label[0], label[1], label[2], label[3]
        box_a = [px-(pw/2), py-(ph/2), px+(pw/2), py+(ph/2)]
        box_b = [lx-(lw/2), ly-(lh/2), lx+(lw/2), ly+(lh/2)]

        # determine the (x, y) of the corners of intersection area
        ax = torch.clamp(box_a[0], min=box_b[0])
        ay = torch.clamp(box_a[1], min=box_b[1])
        bx = torch.clamp(box_a[2], max=box_b[2])
        by = torch.clamp(box_a[3], max=box_b[3])

        # compute the area of intersection
        intersect = torch.abs(torch.clamp((bx - ax), min=0) * torch.clamp((by - ay), min=0))

        # compute the area of both the prediction and ground-truth
        area_a = torch.abs((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = torch.abs((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))

        # compute the iou
        iou = intersect / (area_a + area_b - intersect)
        return iou
 
def get_highest_confidence_box(pred):
    """
    Returns the box with the highest confidence score that has detected an object.
    """
    highest_conf_box = None
    max_confidence = 0

    for box in pred:
        confidence = box[4]  # Assuming confidence score is at index 4
        if confidence > max_confidence:
            highest_conf_box = box
            max_confidence = confidence

    return highest_conf_box
