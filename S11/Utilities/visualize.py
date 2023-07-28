import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchmetrics import ConfusionMatrix
from torchvision import transforms


def plot_class_label_counts(data_loader, classes):
    class_counts = {}
    for class_name in classes:
        class_counts[class_name] = 0
    for _, batch_label in data_loader:
        for label in batch_label:
            class_counts[classes[label.item()]] += 1

    fig = plt.figure()
    plt.suptitle("Class Distribution")
    plt.bar(range(len(class_counts)), list(class_counts.values()))
    plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90)
    plt.tight_layout()
    plt.show()


def plot_data_samples(data_loader, classes):
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()
    plt.suptitle("Data Samples with Labels post Transforms")
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(batch_data[i])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            classes[batch_label[i].item()],
        )

        plt.xticks([])
        plt.yticks([])


def plot_model_training_curves(train_accs, test_accs, train_losses, test_losses):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
    plt.plot()


def plot_confusion_matrix(labels, preds, classes=range(10), normalize=True):
    confmat = ConfusionMatrix(task="multiclass", num_classes=10)
    confmat = confmat(preds, labels).numpy()
    if normalize:
        df_confmat = pd.DataFrame(
            confmat / np.sum(confmat, axis=1)[:, None],
            index=[i for i in classes],
            columns=[i for i in classes],
        )
    else:
        df_confmat = pd.DataFrame(
            confmat,
            index=[i for i in classes],
            columns=[i for i in classes],
        )
    plt.figure(figsize=(7, 5))
    sn.heatmap(df_confmat, annot=True, cmap="Blues", fmt=".3f", linewidths=0.5)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def plot_incorrect_preds(incorrect, classes):
    # incorrect (data, target, pred, output)
    print(f"Total Incorrect Predictions {len(incorrect)}")
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle("Target | Predicted Label")
    for i in range(10):
        plt.subplot(2, 5, i + 1, aspect="auto")

        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(incorrect[i][0])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            f"{classes[incorrect[i][1].item()]}|{classes[incorrect[i][2].item()]}",
            # fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def plot_grad_cam_different_targets(model, loader, classes, device):
    target_layers = [model.layer3[-1]]
    _, (input_tensor, target) = next(enumerate(loader))
    # Get the first image from the batch
    imput_tensor = input_tensor[0].unsqueeze(0).to(device)

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(
        model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
    )

    fig = plt.figure(figsize=(10, 5))
    plt.suptitle(f"GradCAM ID | Target Class : {classes[target[0]]}")
    for i in range(10):
        plt.subplot(2, 5, i + 1, aspect="auto")

        # Get the CAM
        grayscale_cam = cam(
            input_tensor=imput_tensor, targets=[ClassifierOutputTarget(i)]
        )

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(imput_tensor[0, :])
        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"GradCAM {i} | {classes[i]}",
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()


def plot_grad_cam_misclassified(model, incorrect, classes, device):
    target_layers = [model.layer3[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(
        model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()
    )
    fig = plt.figure(figsize=(20, 10))
    for i in range(10):
        misclassified_tuple = incorrect[i]
        input_tensor = misclassified_tuple[0].unsqueeze(0).to(device)
        target_label = misclassified_tuple[1].item()
        predicted_label = misclassified_tuple[2].item()

        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(input_tensor[0, :])

        plt.subplot(4, 5, i * 2 + 1, aspect="auto")
        # Get the CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"P {classes[predicted_label]} | T {classes[target_label]} | Pred CAM ",
        )
        plt.xticks([])
        plt.yticks([])

        plt.subplot(4, 5, i * 2 + 2, aspect="auto")
        # Get the CAM
        grayscale_cam = cam(
            input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_label)]
        )

        # Get the first image from the batch
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            unnormalized.permute(1, 2, 0).cpu().detach().numpy(),
            grayscale_cam,
            use_rgb=True,
            image_weight=0.6,
        )
        plt.imshow(transforms.ToPILImage()(visualization))
        plt.title(
            f"P {classes[predicted_label]} | T {classes[target_label]} | Target CAM ",
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
