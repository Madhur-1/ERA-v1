import numpy as np

from .bert_utils import get_batch


def bert_training_step(
    itr, model, data_loader, batch_iter, loss_fn, optimizer, print_each, device
):
    # Get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)

    masked_input = batch["input_ids"]
    masked_target = batch["target_ids"]

    # Send data to target device
    masked_input, masked_target = masked_input.to(device), masked_target.to(device)

    # Forward pass
    output = model(masked_input, attn_mask=None)

    # Compute the loss
    output_v = output.view(-1, output.size(-1))
    target_v = masked_target.view(-1, 1).squeeze(1)

    loss = loss_fn(output_v, target_v)

    # Backward pass
    loss.backward()
    optimizer.step()

    # Print loss
    if itr % print_each == 0:
        print(
            "it:",
            itr,
            " | loss",
            np.round(loss.item(), 2),
            " | Δw:",
            round(model.token_embed_layer.weight.grad.abs().sum().item(), 3),
        )

    optimizer.zero_grad()
