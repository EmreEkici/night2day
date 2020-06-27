import torch
import torch.nn as nn
from torch.nn import init
#from torchsummary import summary
def define_G():
    generator = UnetGenerator()
    generator.initialize()
    
    return generator

class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()

        block1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        model = [block1]

        block2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 184, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(184),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(184, 304, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(304)
        )
        model += [block2]

        block3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(304, 423, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(423),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(423, 543, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(543),
        )
        model += [block3]

        block4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(543, 663, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(663),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(663, 783, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(783)
        )
        model += [block4]

        block5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(783, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783)
        )
        model += [block5]

        block6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(783, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783)
        )
        model += [block6]

        block7 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(783, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783)
        )
        model += [block7]

        block8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(783, 783, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(783, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783)
        )
        model += [block8]

        block9 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1566, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783),
            nn.Dropout(0.5)
        )
        model += [block9]

        block10 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1566, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783),
            nn.Dropout(0.5)
        )
        model += [block10]

        block11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1566, 783, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(783),
            nn.Dropout(0.5)
        )
        model += [block11]

        block12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1566, 1055, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1055),
            
            nn.ReLU(),
            nn.ConvTranspose2d(1055, 543, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(543)
        )
        model += [block12]

        block13 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1086, 695, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(695),

            nn.ReLU(),
            nn.ConvTranspose2d(695, 304, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(304)
        )
        model += [block13]

        block14 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(608, 336, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(336),

            nn.ReLU(),
            nn.ConvTranspose2d(336, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        model += [block14]

        block15 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        model += [block15]

        self.model = nn.Sequential(*model)

    def initialize(self):
        init.normal_(self.model[0][0].weight)

        for i in range(1, 15):
            block = self.model[i]
            init.normal_(block[1].weight)
            
            if i == 7:
                init.normal_(block[3].weight)
                init.constant_(block[4].weight, 1)
                init.constant_(block[4].bias, 0)
                continue

            if i != 14:
                init.constant_(block[2].weight, 1)
                init.constant_(block[2].bias, 0)


    def generate(self, x):
        model = self.model

        y1 = model[0](x)
        y2 = model[1](y1)
        y3 = model[2](y2)
        y4 = model[3](y3)
        y5 = model[4](y4)
        y6 = model[5](y5)
        y7 = model[6](y6)
        y8 = model[7](y7)
        
        x9 = torch.cat((y8, y7), 1)
        y9 = model[8](x9)
        x10 = torch.cat((y9, y6), 1)
        y10 = model[9](x10)
        x11 = torch.cat((y10, y5), 1)
        y11 = model[10](x11)
        x12 = torch.cat((y11, y4), 1)
        y12 = model[11](x12)
        x13 = torch.cat((y12, y3), 1)
        y13 = model[12](x13)
        x14 = torch.cat((y13, y2), 1)
        y14 = model[13](x14)
        x15 = torch.cat((y14, y1), 1)
        y = model[14](x15)

        return y

#g = UnetGenerator()
#print(g)