import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import Yolov7
from dataset import DataSetProcessorTrainingVal, DataSetProcessorTest
from torchvision import transforms
from yolov7.utils.loss import smooth_BCE, FocalLoss, is_parallel, bbox_iou
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                #print(iou)
                lbox += (1.0 - iou).mean()  # iou loss
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    #t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach(), iou

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            #print(p[i].shape, "predicts")
            anchors = self.anchors[i]
            #print()
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            new_targets = targets.clone().detach()
            # Ensure gain does not cause gradient computation issues
            t = new_targets * gain.float().detach().requires_grad_()
            
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class Train:
    
    def __init__(self):
        # Varibles
        BATCH_SIZE = 32
        self.epochs = 8
        LEARNING_RATE = 0.001

        # Paths
        train_dir = 'recognition/Data/ISIC2018_Task1-2_Training_Input_x2'
        annotation_dir = 'recognition/Data/train_labels'
        val_dir = 'recognition/Data/ISIC2018_Task1_Training_GroundTruth_x2'
        test_dir = 'recognition/Data/ISIC2018_Task1-2_Test_Input'
        # function that transforms the data
        transform_train = transforms.Compose([
            transforms.Resize((640, 640)),  # Resize image to 128x128
            transforms.ToTensor(),            # Convert PIL Image to Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        transform_val = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Ensure these match training
        ])

        # set up Data change
        train = DataSetProcessorTrainingVal(train_dir, annotation_dir, transform_train)
        validation = DataSetProcessorTrainingVal(val_dir, annotation_dir, transform_val)
        test = DataSetProcessorTest(test_dir, transform_train)

        self.training_dataset = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        self.validation_dataset = DataLoader(validation, batch_size=BATCH_SIZE, shuffle=False)
        self.test_dataset = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Yolov7()
        self.optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.model = model.to(self.device)

        # Define and attach hyperparameters
        hyp = {
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0
        }

        yolo_model = model.model

        if hasattr(yolo_model, 'model'):
            yolo_model = yolo_model.model
        yolo_model.hyp = hyp
        yolo_model.gr = 1.0
        self.loss_function = ComputeLoss(yolo_model)

    def train_one_epoch(self, model, training_dataset, optimizer, loss_function, device):
        model.train()
        print("one epoch")
        i = 0
        running_loss = 0
        for image, label in training_dataset:
            print("training loop", i)
            image = image.to(device)
            label = label.to(device)

            prediction = model(image)
            
            loss, _, _= loss_function(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss for this batch
            running_loss = running_loss + loss.item()     
            i+=1
        
        avg_loss = running_loss / i

        return avg_loss

    def calculate_accuracy(self, iou_values, threshold=0.8):
        if not isinstance(iou_values, torch.Tensor):
            iou_values = torch.tensor(iou_values)
        correct_predictions = (iou_values >= threshold).sum().item()
        total_predictions = len(iou_values)

        accuracy = correct_predictions / total_predictions * 100
        return accuracy

    def load_checkpoint(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def evaluate(self, checkpoint_path='model_checkpoint.pth'):
        self.model.eval()
        iou_scores = []
        i = 0
        try:
            self.load_checkpoint(self.model, self.optimizer, checkpoint_path)
            print("Checkpoint loaded.")
        except FileNotFoundError:
            print(f"No checkpoint found at {checkpoint_path}. Please train the model first.")
            return
        

        with torch.no_grad():
            for images, labels in self.validation_dataset:
                print("Validation loop", i)
                images = images.to(self.device)
                label = labels.to(self.device)
                prediction = self.model(images)
        
                _, _, IOU = self.loss_function(prediction[1], label)
                average_iou = IOU.mean().item()
                iou_scores.append(average_iou)
                i+=1

        average_iou = np.mean(iou_scores)
        accuracy = self.calculate_accuracy(iou_scores) if iou_scores else 0
        print(f"Average IoU: {average_iou:.4f}, Accuracy: {accuracy:.4f}")
        return average_iou, accuracy


    def save(self, model, optimizer, epoch, path):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, path)

    def start_training(self):
        train_loss_list = []
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(self.model, self.training_dataset, self.optimizer, self.loss_function, self.device)
            print(train_loss)
            train_loss_list.append(train_loss)
            self.save(self.model, self.optimizer, epoch, 'model_checkpoint.pth')
        return train_loss_list, self.model, self.validation_dataset, self.device

        


