from mimetypes import init
import torch
import numpy as np
import torch.nn as nn

from config import *

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.args.latent_dimension, 128, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        output = self.model(z)
        return output

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.args = args

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(-1, 1).squeeze(1)
        return output

def main(args):
    g = Generator(args)
    print(torch.randn((4, args.latent_dimension)).shape)
    tmp = torch.randn((4, args.latent_dimension)).view(-1, args.latent_dimension, 1, 1)
    print(tmp.shape)
    g_out = g(tmp)
    d = Discriminator(args)
    d_out = d(g_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters parser', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)