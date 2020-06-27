import torch
import torch.nn as nn
from torch.nn import init

def define_D():
    discriminator = NLayerDiscriminator()
    discriminator.initialize()

    return discriminator

class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super(NLayerDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),

            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def initialize(self):
        init.normal_(self.model[0].weight)
        
        init.normal_(self.model[2].weight)
        init.constant_(self.model[3].weight, 1)
        init.constant_(self.model[3].bias, 0)

        init.normal_(self.model[5].weight)
        init.constant_(self.model[6].weight, 1)
        init.constant_(self.model[6].bias, 0)

        init.normal_(self.model[8].weight)
        init.constant_(self.model[9].weight, 1)
        init.constant_(self.model[9].bias, 0)

        init.normal_(self.model[11].weight)
    
    def forward(self, input):
        return self.model(input)