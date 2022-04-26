import torch.nn as nn


class ResnetHead(nn.Module):

    def __init__(self, channel, num_classes):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel, num_classes)

    def forward(self, x):
        output = self.avg_pool(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


class VGGHead(nn.Module):

    def __init__(self, channel, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(channel, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        output = self.classifier(x)
        return output
