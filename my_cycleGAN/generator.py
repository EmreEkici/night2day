import torch
import torch.nn as nn
from torch.nn import init

def define_G(num_res_blocks=9):
    generator = ResnetGenerator(num_res_blocks)
    generator.initialize()
    
    return generator

class ResnetGenerator(nn.Module):
    def __init__(self, num_res_blocks):
        super(ResnetGenerator, self).__init__()
        self.num_res_blocks = num_res_blocks

        block_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        model = [block_conv]

        block_down = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        model += [block_down]

        for n in range(num_res_blocks):
            block_res = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(256, 256, 3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(256)
            )
            model += [block_res]

        block_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        model += [block_up]

        block_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        model += [block_conv]

        self.model = nn.Sequential(*model)

    def initialize(self):
        init.normal_(self.model[0][1].weight)
        init.constant_(self.model[0][2].weight, 1)
        init.constant_(self.model[0][2].bias, 0)

        init.normal_(self.model[1][0].weight)
        init.constant_(self.model[1][1].weight, 1)
        init.constant_(self.model[1][1].bias, 0)
        
        init.normal_(self.model[1][3].weight)
        init.constant_(self.model[1][4].weight, 1)
        init.constant_(self.model[1][4].bias, 0)

        for n in range(2, self.num_res_blocks + 2):
            init.normal_(self.model[n][1].weight)
            init.constant_(self.model[n][2].weight, 1)
            init.constant_(self.model[n][2].bias, 0)

            init.normal_(self.model[n][5].weight)
            init.constant_(self.model[n][6].weight, 1)
            init.constant_(self.model[n][6].bias, 0)

        init.normal_(self.model[self.num_res_blocks + 2][0].weight)
        init.constant_(self.model[self.num_res_blocks + 2][1].weight, 1)
        init.constant_(self.model[self.num_res_blocks + 2][1].bias, 0)
        
        init.normal_(self.model[self.num_res_blocks + 2][3].weight)
        init.constant_(self.model[self.num_res_blocks + 2][4].weight, 1)
        init.constant_(self.model[self.num_res_blocks + 2][4].bias, 0)

        init.normal_(self.model[self.num_res_blocks + 3][1].weight)

    def forward(self, x):
        y_conv = self.model[0](x)
        y_down = self.model[1](y_conv)

        x_res = y_down
        for n in range(self.num_res_blocks):
            y_res = self.model[n + 2](x_res)
            x_res = y_res + x_res
        y_res = x_res

        y_up = self.model[self.num_res_blocks + 2](y_res)
        y = self.model[self.num_res_blocks + 3](y_up)

        return y