import numpy as np


def get_batch(loader, loader_iter):
    """
    Get a batch of data from a PyTorch DataLoader instance.
    """
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


def save_model_embeddings(model, dataset, embeddings_path, embeddings_name_path):
    """
    Save model embeddings to a file.
    """
    print("Saving model embeddings to", embeddings_path)
    N = 3000
    np.savetxt(
        embeddings_path,
        np.round(model.token_embed_layer.weight.detach().cpu().numpy()[:N], 2),
        delimiter="\t",
        fmt="%1.2f",
    )
    s = [dataset.rvocab[i] for i in range(N)]
    open(embeddings_name_path, "w").write("\n".join(s))

    print("Saved model embeddings!")
