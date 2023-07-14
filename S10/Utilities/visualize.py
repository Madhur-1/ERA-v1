import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
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
