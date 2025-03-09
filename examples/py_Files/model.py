# alexnet
def install_alexnet():
    from torchvision import models

    alexnet = models.alexnet(True)


if __name__ == "__main__":
    install_alexnet()
