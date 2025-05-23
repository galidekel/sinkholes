import torch
from unet_parts import *
import matplotlib.pyplot as plt


class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.output_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # B x heads x HW x head_dim
        K = self.key(x).view(B, self.num_heads, self.head_dim, H * W)  # B x heads x head_dim x HW
        V = self.value(x).view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)  # B x heads x HW x head_dim

        attn = torch.matmul(Q, K) / (self.head_dim ** 0.5)  # Scaled dot-product attention
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # B x heads x HW x head_dim
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)  # Merge heads back

        out = self.output_proj(out)
        return self.gamma * out + x  # Residual connection



class SelfAttention2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        Q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        K = self.key(x).view(B, -1, H * W)
        V = self.value(x).view(B, -1, H * W)

        attn = torch.bmm(Q, K)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,add_attn = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.add_attn = add_attn
        if self.add_attn:
            self.attn = SelfAttention2D(1024)
            self.attn = MultiHeadSelfAttention2D(1024,8)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.add_attn:
            x5=self.attn(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def get_num_params(self):
        full_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return full_sum

if __name__ == '__main__':
    unet = UNet(1,1)
    unet_size = unet.get_num_params()
    print(unet_size)
