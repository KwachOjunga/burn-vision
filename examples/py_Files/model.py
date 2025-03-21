# alexnet
def install_models():
    from torchvision import models

    # alexnet = models.alexnet(True)
    googlenet = models.googlenet(True)
    inception = models.inception_v3(True)
    densenet = models.densenet121(True)
    maxvit = models.maxvit_t(True)
    regnet = models.regnet_x_16gf(True)
    efficientnet = models.efficientnet_b0(True)
    mnasnet = models.mnasnet0_5(True)
    vgg = models.vgg11(True)
    vision_transformer = models.vit_b_16(True)


if __name__ == "__main__":
    install_models()
