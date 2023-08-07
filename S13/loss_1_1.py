"""
Implementation of Yolo Loss Function similar to the one in Yolov3 paper,
the difference from what I can tell is I use CrossEntropy for the classes
instead of BinaryCrossEntropy.
"""
import random
import torch
import torch.nn as nn
import config
from utils import intersection_over_union

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target):
        combined_loss = 0
        for i in range(3):
            # Check where obj and noobj (we ignore if target[i] == -1)
            obj = target[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = target[i][..., 0] == 0  # in paper this is Inoobj_i

            # ======================= #
            #   FOR NO OBJECT LOSS    #
            # ======================= #

            no_object_loss = self.bce(
                (predictions[i][..., 0:1][noobj]), (target[i][..., 0:1][noobj]),
            )

            # ==================== #
            #   FOR OBJECT LOSS    #
            # ==================== #

            anchors = scaled_anchors[i]

            anchors = anchors.reshape(1, 3, 1, 1, 2)
            box_preds = torch.cat([self.sigmoid(predictions[i][..., 1:3]), torch.exp(predictions[i][..., 3:5]) * anchors], dim=-1)
            ious = intersection_over_union(box_preds[obj], target[i][..., 1:5][obj]).detach()
            object_loss = self.mse(self.sigmoid(predictions[i][..., 0:1][obj]), ious * target[i][..., 0:1][obj])

            # ======================== #
            #   FOR BOX COORDINATES    #
            # ======================== #

            predictions[i][..., 1:3] = self.sigmoid(predictions[i][..., 1:3])  # x,y coordinates
            target[i][..., 3:5] = torch.log(
                (1e-16 + target[i][..., 3:5] / anchors)
            )  # width, height coordinates
            box_loss = self.mse(predictions[i][..., 1:5][obj], target[i][..., 1:5][obj])

            # ================== #
            #   FOR CLASS LOSS   #
            # ================== #

            class_loss = self.entropy(
                (predictions[i][..., 5:][obj]), (target[i][..., 5][obj].long()),
            )

            print("__________________________________")
            print(self.lambda_box * box_loss)
            print(self.lambda_obj * object_loss)
            print(self.lambda_noobj * no_object_loss)
            print(self.lambda_class * class_loss)
            print("\n")
            combined_loss+= self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss

        return combined_loss