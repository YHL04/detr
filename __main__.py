

from models import DETR
from losses import SetCriterion


def train_one_epoch(model, criterion, dataloader, optimizer, device, epoch, max_norm=0):
    """
    """
    model.train()

    for i, batch in enumerate(dataloader):
        samples, targets = batch

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def main(epochs=10):

    model = DETR()
    criterion = SetCriterion()
    dataloader = None
    device = None

    for epoch in range(epochs):
        train_one_epoch(model, criterion, dataloader, device, epoch)


if __name__ == "__main__":
    main()

