import torch

test_incorrect_pred = {"images": [], "ground_truths": [], "predicted_vals": []}


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    scheduler,
    train_losses,
    train_acc,
):
    model.train()
    # tqdm._instances.clear()
    # pbar = tqdm(train_loader, position=0, leave=True, ascii=True)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        print(
            f"Train: {((batch_idx+1)/len(train_loader))*100:0.0f}% Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}",
            end="\r",
            flush=True,
        )
    # pbar.close()
    # del pbar
    print(
        f"Train: {((batch_idx+1)/len(train_loader))*100:0.0f}% Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}",
    )
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))
    return train_acc, train_losses


def test(model, device, test_loader, criterion, test_losses, test_acc):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_acc, test_losses


def get_all_and_incorrect_preds(model, loader, device):
    incorrect = []
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1).cpu()
            target = target.cpu()
            all_preds = torch.cat((all_preds, pred), dim=0)
            all_labels = torch.cat((all_labels, target), dim=0)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append((d.cpu(), t, p, o[p.item()].cpu()))

    return all_preds, all_labels, incorrect
