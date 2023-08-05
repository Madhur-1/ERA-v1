import matplotlib.pyplot as plt
from torchvision import transforms


def plot_incorrect_preds(incorrect, classes, num_imgs):
    import random

    # num_imgs is a multiple of 5
    assert num_imgs % 5 == 0
    assert len(incorrect) >= num_imgs

    incorrect_inds = random.sample(range(len(incorrect)), num_imgs)

    # incorrect (data, target, pred, output)
    fig = plt.figure(figsize=(10, num_imgs // 2))
    plt.suptitle("Target | Predicted Label")
    for i in range(num_imgs):
        curr_incorrect = incorrect[incorrect_inds[i]]
        plt.subplot(num_imgs // 5, 5, i + 1, aspect="auto")

        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(curr_incorrect[0])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            f"{classes[curr_incorrect[1].item()]}|{classes[curr_incorrect[2].item()]}",
            # fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    return fig
