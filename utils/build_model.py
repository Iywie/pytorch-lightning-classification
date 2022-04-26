from PL_Modules.lit_resnet import LitResnet
from PL_Modules.lit_vgg import LitVGG


CONFIGS = {
    'resnet': LitResnet,
    'vgg': LitVGG,
}


def build_model(model_name):
    return CONFIGS[model_name]
