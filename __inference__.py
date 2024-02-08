

import torch

from detr import DETR


def inference(detr, x):
    """

    Parameters:
        detr (DETR): Trained model to be used for inference
        x (np.ndarray): Input to be used for object detection of size (B, C, H, w)

    Returns:
        boxes (np.ndarray): 

    """
    classes, boxes = detr(x)
    print(classes.shape)
    print(boxes.shape)


def main():
    detr = DETR(H=512, W=512, num_classes=10)
    x = torch.zeros((1, 3, 512, 512))
    inference(detr, x)


if __name__ == "__main__":
    main()

