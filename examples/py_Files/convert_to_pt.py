import torch
import sys
import torchvision.models as models

WORK_DIR = sys.argv[2]


# this needs to get more general prurpose than this otherwise it becomees hyper specific
def convert():
    """
    this function converts a model with a .pth extension
    to .pt to allow it to be handled by burn_vision
    """
    try:
        name = sys.argv[1]
        model = models.alexnet()
        state = torch.load(name)
        model.load_state_dict(state)
        torch.save(model, f"{WORK_DIR}/models/alexnet.pt")
        print("Model conversion complete")
    except Exception as e:
        raise e


if __name__ == "__main__":
    convert()
