from torch.nn import Module, Sequential, Conv2d, ReLU, Linear, Sigmoid

def get_conv2d_relu_layer(*args):
    return Sequential(
        Conv2d(*args),
        ReLU()
    )

class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 =get_conv2d_relu_layer(3, 8, 5, 1, 2)
        self.conv2 = get_conv2d_relu_layer(8, 16, 5, 1, 2)
        self.conv3 = get_conv2d_relu_layer(16, 64, 5, 1, 2)
        self.conv4 = get_conv2d_relu_layer(64, 128, 5, 1, 2)
        self.conv5 = get_conv2d_relu_layer(128, 128, 5, 1, 2)
        self.conv6 = get_conv2d_relu_layer(128, 128, 5, 1, 2)
        self.conv_mask = Conv2d(128, 1, 5, 1, 2)
        self.conv7 = get_conv2d_relu_layer(128, 64, 5, 4, 1)
        self.conv8 = get_conv2d_relu_layer(64, 32, 5, 4, 1)
        self.fc = Sequential(
            Linear(32 * 14 * 14, 1024),
            Linear(1024, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        mask = self.conv_mask(x)
        x = self.conv7(x * mask)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), -1)
        return mask, self.fc(x)