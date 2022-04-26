from PL_DataModules.cifar10 import CIFAR10DataModule


CONFIGS = {
    'cifar10': CIFAR10DataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
