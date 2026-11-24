from torch import nn


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners


    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

        return x


class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation):
        super().__init__()

        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)


        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)


        out = self.activation(out)
        out = self.conv2(out)


        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()

        self.align_corners = align_corners
        self.groups = 1
        out_features = features


        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation)
        self.skip_add = nn.quantized.FloatFunctional()


    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        output = self.out_conv(output)

        return output