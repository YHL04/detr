

from detr import DETR
from losses import SetCriterion

from engine import train_one_epoch


def main(epochs=10):

    model = DETR(H=512, W=512, num_classes=10)
    criterion = SetCriterion()
    dataloader = None
    device = None

    for epoch in range(epochs):
        train_one_epoch(model, criterion, dataloader, device, epoch)


if __name__ == "__main__":
    main()

