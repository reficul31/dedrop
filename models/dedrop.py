from torch.nn import Module

from .vae import VAE
from .inpaint import InpaintNet

class DedropNet(Module):
    def __init__(self):
        super(DedropNet, self).__init__()
        self.vae = VAE()
        self.inpaint = InpaintNet()
    
    def forward(self, image):
        mask, _, _, _ = self.vae(image)
        fame1, frame2, image = self.inpaint(image, mask)
        return image, fame1, frame2
