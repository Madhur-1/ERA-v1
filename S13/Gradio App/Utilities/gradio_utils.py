import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from Utilities.model import YOLOv3
from Utilities.transforms import test_transforms
from Utilities.utils import cells_to_bboxes, non_max_suppression, plot_image


def plot_bboxes(
    input_img,
    model,
    thresh=0.6,
    iou_thresh=0.5,
    anchors=None,
):
    input_img = test_transforms(image=input_img)["image"]
    input_img = input_img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(input_img)
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(out[i], anchor, S=S, is_preds=True)
            bboxes = boxes_scale_i[0]

    nms_boxes = non_max_suppression(
        bboxes,
        iou_threshold=iou_thresh,
        threshold=thresh,
        box_format="midpoint",
    )
    fig = plot_image(input_img[0].permute(1, 2, 0).detach().cpu(), nms_boxes)
    return fig, input_img


def return_top_objectness_class_preds(model, input_img, gradcam_output_stream):
    out = model(input_img)[gradcam_output_stream]

    # Step 1: Extract objectness scores
    objectness_scores = out[..., 0]

    # Step 2: Get the index of the highest objectness score
    max_obj_arg = torch.argmax(objectness_scores)

    max_obj_arg_onehot = torch.zeros(objectness_scores.flatten().shape[0])
    max_obj_arg_onehot[max_obj_arg] = 1

    max_obj_arg_onehot = max_obj_arg_onehot.reshape_as(
        objectness_scores,
    ).int()

    selected_elements = out[max_obj_arg_onehot == 1]
    selected_elements = selected_elements[:, 5:]

    return selected_elements


class TopObjectnessClassPreds(pl.LightningModule):
    def __init__(self, model, gradcam_output_stream):
        super().__init__()
        self.model = model
        self.gradcam_output_stream = gradcam_output_stream

    def forward(self, x):
        return return_top_objectness_class_preds(
            self.model, x, self.gradcam_output_stream
        )


def generate_gradcam_output(org_img, model, input_img, gradcam_output_stream: int = 0):
    TopObjectnessClassPredsObj = TopObjectnessClassPreds(model, gradcam_output_stream)
    gradcam_model_layer = [15, 22, 29]
    cam = GradCAM(
        model=TopObjectnessClassPredsObj,
        target_layers=[
            TopObjectnessClassPredsObj.model.layers[
                gradcam_model_layer[gradcam_output_stream]
            ]
        ],
    )
    grayscale_cam = cam(input_tensor=input_img, targets=None)
    grayscale_cam = np.sum(grayscale_cam, axis=-1)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(
        org_img / 255,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5,
    )
    return visualization
