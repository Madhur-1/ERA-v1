import matplotlib.pyplot as plt


def plot_class_label_counts(data_loader):
    class_counts = {}
    for _, batch_label in data_loader:
        for label in batch_label:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

    fig = plt.figure()
    plt.bar(range(len(class_counts)), list(class_counts.values()))
    plt.xticks(range(len(class_counts)), list(class_counts.keys()))
    plt.show()


def plot_data_samples(data_loader):
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap="gray")
        plt.title(batch_label[i].item())
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
