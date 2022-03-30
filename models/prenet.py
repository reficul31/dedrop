import torch

from torch.autograd import Variable
from torch.nn.functional import relu
from torch.nn import Module, Sequential, Conv2d, ReLU, Sigmoid, Tanh

class PReNet(Module):
    def __init__(self, recurrent_iter=6):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = Sequential(
            Conv2d(6, 32, 3, 1, 1),
            ReLU()
            )
        self.res_conv1 = Sequential(
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU()
            )
        self.res_conv2 = Sequential(
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU()
            )
        self.res_conv3 = Sequential(
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU()
            )
        self.res_conv4 = Sequential(
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU()
            )
        self.res_conv5 = Sequential(
            Conv2d(32, 32, 3, 1, 1),
            ReLU(),
            Conv2d(32, 32, 3, 1, 1),
            ReLU()
            )
        self.conv_i = Sequential(
            Conv2d(32 + 32, 32, 3, 1, 1),
            Sigmoid()
            )
        self.conv_f = Sequential(
            Conv2d(32 + 32, 32, 3, 1, 1),
            Sigmoid()
            )
        self.conv_g = Sequential(
            Conv2d(32 + 32, 32, 3, 1, 1),
            Tanh()
            )
        self.conv_o = Sequential(
            Conv2d(32 + 32, 32, 3, 1, 1),
            Sigmoid()
            )
        self.conv = Sequential(
            Conv2d(32, 3, 3, 1, 1),
            )

    def forward(self, input):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, row, col)).to(device)
        c = Variable(torch.zeros(batch_size, 32, row, col)).to(device)

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = relu(self.res_conv1(x) + resx)
            resx = x
            x = relu(self.res_conv2(x) + resx)
            resx = x
            x = relu(self.res_conv3(x) + resx)
            resx = x
            x = relu(self.res_conv4(x) + resx)
            resx = x
            x = relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list
