import torch
import torch.nn as nn
from torch.nn import init

def define_G():
    generator = UnetGenerator()
    generator.initialize()
    
    return generator

class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()

        block1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=False)
        )
        model = [block1]

        block2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        model += [block2]

        block3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        model += [block3]

        block4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        model += [block4]

        block5 = nn.Sequential( 
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block5]

        block6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        model += [block6]

        block7 = nn.Sequential( 
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        model += [block7]

        block8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        model += [block8]

        block9 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        model += [block9]

        block10 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(2048, 2048, 4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048)
        )
        model += [block10]

        block11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4096, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.Dropout(0.5)
        )
        model += [block11]

        block12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4096, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.Dropout(0.5)
        )
        model += [block12]

        block13 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4096, 2048, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.Dropout(0.5)
        )
        model += [block13]

        block14 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(4096, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block14]

        block15 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        model += [block15]

        block16 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        model += [block16]

        block17 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        model += [block17]

        block18 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        model += [block18]

        block19 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        model += [block19]

        self.model = nn.Sequential(*model)

    def initialize(self):
        init.normal_(self.model[0][0].weight)

        for i in range(1, 19):
            block = self.model[i]
            init.normal_(block[1].weight)
            
            if i == 9:
                init.normal_(block[3].weight)
                init.constant_(block[4].weight, 1)
                init.constant_(block[4].bias, 0)
                continue

            if i != 18:
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
        y9 = model[8](y8)
        y10 = model[9](y9)
        
        x11 = torch.cat((y10, y9), 1)
        y11 = model[10](x11)
        x12 = torch.cat((y11, y8), 1)
        y12 = model[11](x12)
        x13 = torch.cat((y12, y7), 1)
        y13 = model[12](x13)
        x14 = torch.cat((y13, y6), 1)
        y14 = model[13](x14)
        x15 = torch.cat((y14, y5), 1)
        y15 = model[14](x15)
        x16 = torch.cat((y15, y4), 1)
        y16 = model[15](x16)
        x17 = torch.cat((y16, y3), 1)
        y17 = model[16](x17)
        x18 = torch.cat((y17, y2), 1)
        y18 = model[17](x18)
        x19 = torch.cat((y18, y1), 1)
        y = model[18](x19)

        return y

#g = UnetGenerator()
#print(g)