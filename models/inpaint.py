import torch

from torch.nn import Module, Conv2d, ReLU, Sequential, ConvTranspose2d, ReflectionPad2d, AvgPool2d

def get_conv2d_relu_layer(*args, **kwargs):
    return Sequential(
        Conv2d(*args, **kwargs),
        ReLU()
    )

class InpaintNet(Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.conv1 = get_conv2d_relu_layer(6, 64, 5, 1, 2)
        self.conv2 = get_conv2d_relu_layer(64, 128, 3, 2, 1)
        self.conv3 = get_conv2d_relu_layer(128, 128, 3, 1, 1)
        self.conv4 = get_conv2d_relu_layer(128, 256, 3, 2, 1)
        self.conv5 = get_conv2d_relu_layer(256, 256, 3, 1, 1)
        self.conv6 = get_conv2d_relu_layer(256, 256, 3, 1, 1)
        self.diconv1 = get_conv2d_relu_layer(256, 256, 3, 1, 2, dilation = 2)
        self.diconv2 = get_conv2d_relu_layer(256, 256, 3, 1, 4, dilation = 4)
        self.diconv3 = get_conv2d_relu_layer(256, 256, 3, 1, 8, dilation = 8)
        self.diconv4 = get_conv2d_relu_layer(256, 256, 3, 1, 16, dilation = 16)
        self.conv7 = get_conv2d_relu_layer(256, 256, 3, 1, 1)
        self.conv8 = get_conv2d_relu_layer(256, 256, 3, 1, 1)
        self.conv9 = get_conv2d_relu_layer(128, 128, 3, 1, 1)
        self.conv10 = get_conv2d_relu_layer(64, 32, 3, 1, 1)
        self.outframe1 = get_conv2d_relu_layer(256, 3, 3, 1, 1)
        self.outframe2 = get_conv2d_relu_layer(128, 3, 3, 1, 1)
        self.deconv1 = Sequential(
            ConvTranspose2d(256, 128, 4, 2, 1),
            ReflectionPad2d((1, 0, 1, 0)),
            AvgPool2d(2, stride = 1),
            ReLU()
        )
        self.deconv2 = Sequential(
            ConvTranspose2d(128, 64, 4, 2, 1),
            ReflectionPad2d((1, 0, 1, 0)),
            AvgPool2d(2, stride = 1),
            ReLU()
        )
        self.output = Conv2d(32, 3, 3, 1, 1)

    def forward(self, input, mask):
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return frame1, frame2, x