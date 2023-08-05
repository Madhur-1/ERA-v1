import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from . import config
from .transforms import test_transforms


def generate_confidences(
    model,
    input_img,
    num_top_preds,
):
    input_img = test_transforms(image=input_img)
    input_img = input_img["image"]

    input_img = input_img.unsqueeze(0)
    model.eval()
    log_probs = model(input_img)[0].detach()
    model.train()
    probs = torch.exp(log_probs)

    confidences = {
        config.CLASSES[i]: float(probs[i]) for i in range(len(config.CLASSES))
    }
    # Select top 5 confidences based on value
    confidences = {
        k: v
        for k, v in sorted(confidences.items(), key=lambda item: item[1], reverse=True)[
            :num_top_preds
        ]
    }
    return input_img, confidences


def generate_gradcam(
    model,
    org_img,
    input_img,
    show_gradcam,
    gradcam_layer,
    gradcam_opacity,
):
    if show_gradcam:
        if gradcam_layer == -1:
            target_layers = [model.l3[-1]]
        elif gradcam_layer == -2:
            target_layers = [model.l2[-1]]

        cam = GradCAM(
            model=model,
            target_layers=target_layers,
        )
        grayscale_cam = cam(input_tensor=input_img, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            org_img / 255,
            grayscale_cam,
            use_rgb=True,
            image_weight=(1 - gradcam_opacity),
        )
    else:
        visualization = None
    return visualization


def generate_missclassified_imgs(
    model,
    show_misclassified,
    num_misclassified,
):
    if show_misclassified:
        plot = model.plot_incorrect_predictions_helper(num_misclassified)
    else:
        plot = None
    return plot
