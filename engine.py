


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, max_norm=0):
    """
    """
    model.train()
    criterion.train()

    for i, batch in enumerate(dataloader):
        samples, targets = batch

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()