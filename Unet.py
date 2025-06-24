import torch
import torch.nn as nn
import torchvision

class UNetResNet(nn.Module):
    def __init__(self, n_classes, out_size=(128, 128)):
        super().__init__()
        base_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
        self.base_layers = list(base_model.children())
        self.encoder1 = nn.Sequential(*self.base_layers[:3])
        self.encoder2 = nn.Sequential(*self.base_layers[3:5])
        self.encoder3 = self.base_layers[5]
        self.encoder4 = self.base_layers[6]
        self.encoder5 = self.base_layers[7]

        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up5 = self.up_block(512, 256)
        self.up4 = self.up_block(256, 128)
        self.up3 = self.up_block(128, 64)
        self.up2 = self.up_block(64, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

        # Upsample to match input/mask size
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)

    def up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        center = self.center(e5)
        d5 = self.up5(center) + e4
        d4 = self.up4(d5) + e3
        d3 = self.up3(d4) + e2
        d2 = self.up2(d3) + e1
        out = self.final(d2)
        out = self.upsample(out)  # Upsample logits to input/mask size
        return out
