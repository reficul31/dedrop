
from .vae import VAE
from .inpaint import InpaintNet
from torch.nn import Module
from torch.nn.functional import softmax

class DedropNet(Module):
    def __init__(self):
        super(DedropNet, self).__init__()
        self.vae = VAE()
        self.inpaint = InpaintNet()
    
    def forward(self, image):
        mask = self.vae(image)
        mask = softmax(mask)
        
        mod = image - mask

        image, mask = self.inpaint(mod, mask)
        return image, mask