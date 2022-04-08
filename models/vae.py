from torch.autograd import Variable
from torch.nn import Module, Conv2d, BatchNorm2d, Linear, BatchNorm1d, ReLU, ConvTranspose2d

class VAE(Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.conv2 = Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = BatchNorm2d(32)
        self.conv3 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = BatchNorm2d(64)
        self.conv4 = Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = BatchNorm2d(16)

        # Latent vectors mu and sigma
        self.fc1 = Linear(25 * 25 * 16, 2048)
        self.fc_bn1 = BatchNorm1d(2048)
        self.fc21 = Linear(2048, 2048)
        self.fc22 = Linear(2048, 2048)

        # Sampling vector
        self.fc3 = Linear(2048, 2048)
        self.fc_bn3 = BatchNorm1d(2048)
        self.fc4 = Linear(2048, 25 * 25 * 16)
        self.fc_bn4 = BatchNorm1d(25 * 25 * 16)

        # Decoder
        self.conv5 = ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = BatchNorm2d(64)
        self.conv6 = ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = BatchNorm2d(32)
        self.conv7 = ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = BatchNorm2d(16)
        self.conv8 = ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 25 * 25 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        
        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 25, 25)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 3, 100, 100)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar