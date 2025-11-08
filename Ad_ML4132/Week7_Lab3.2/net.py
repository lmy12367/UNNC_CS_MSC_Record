import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()

        self.initial_conv = DoubleConv(in_channels, features[0])

        for i in range(len(features) - 1):
            self.downs.append(
                nn.Sequential(
                    nn.MaxPool2d(2),
                    DoubleConv(features[i], features[i + 1])
                )
            )

    def forward(self, x):
        skip_connections = []

        x = self.initial_conv(x)
        skip_connections.append(x)

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)

        return x, skip_connections


class Decoder(nn.Module):
    def __init__(self, features=[512, 256, 128, 64]):
        super().__init__()
        self.ups = nn.ModuleList()

        for feature in features[:-1]:
            self.ups.append(
                nn.ConvTranspose2d(feature, feature // 2, kernel_size=2, stride=2)
            )

        self.decoder_convs = nn.ModuleList()

        for feature in features[:-1]:
            self.decoder_convs.append(DoubleConv(feature, feature // 2))

    def forward(self, x, skip_connections):

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x)

            skip = skip_connections[idx + 1]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat((skip, x), dim=1)
            x = self.decoder_convs[idx](x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features[::-1])
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        x = self.decoder(bottleneck, skip_connections)
        return torch.sigmoid(self.final_conv(x))


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1).to(device)
    print(model)

    x = torch.randn((1, 3, 256, 256)).to(device)
    print(f"Input shape: {x.shape}")

    pred = model(x)
    print(f"Output shape: {pred.shape}")

    assert pred.shape == (1, 1, 256, 256)
    print("Model output shape is correct!")