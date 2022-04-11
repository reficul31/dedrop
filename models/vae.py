from torch.autograd import Variable
from torch.nn import Module, Conv2d, Linear, ReLU, ConvTranspose2d, Sequential

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Encoder(Module):
    def __init__(self, nc, nef, nz):
        super(Encoder, self).__init__()
        self.main = Sequential(
            Conv2d(nc, nef, 4, 2, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(nef*8, nef*16, 4, 1),
            ReLU(True),
            View((-1, nef*16 * 1 * 1)),
            Linear(nef*16, nz* 2),
        )
    
    def forward(self, input):
        distributions = self.main(input)
        return distributions

class Decoder(Module):
    def __init__(self, nz, nef, nc):
        super(Decoder, self).__init__()
        self.main = Sequential(
            Linear(nz, nef*16),
            View((-1, nef*16, 1, 1)),
            ReLU(True),
            ConvTranspose2d(nef*16, nef * 8, 4, 1, 0, bias=False),
            ReLU(True),
            ConvTranspose2d(nef * 8, nef * 4, 4, 2, 1, bias=False),
            ReLU(True),
            ConvTranspose2d(nef * 4, nef * 2, 4, 2, 1, bias=False),
            ReLU(True),
            ConvTranspose2d(nef * 2, nef, 4, 2, 1, bias=False),
            ReLU(True),
            ConvTranspose2d(nef, nc, 4, 2, 1, bias=False),
            ReLU(True)
        )

    def forward(self, input):
        R = self.main(input)
        return R

class VAE(Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.nc = 3
        self.nz = 128
        self.nef= 32
        self.encoder = Encoder(self.nc, self.nef, self.nz)
        self.decoder = Decoder(self.nz, self.nef, 1)

    def sample (self, input):
        return  self.decoder(input)

    def forward(self, input):
        distributions = self.encoder(input)
        mu = distributions[:, :self.nz]
        logvar = distributions[:, self.nz:]
        z = reparametrize(mu, logvar)
        R = self.decoder(z)
        return  R, mu,logvar, z