import torch
import torch.nn as nn


def single_conv(in_channels, out_channels):
    """
    Create a single 3x3 convolution layer + ReLU activation module
    following the original U-Net architecture.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True)  # inplace=True saves memory
    )


def double_conv(in_channels, out_channels):
    """
    Create a double 3x3 convolution layer + ReLU activation module
    following the original U-Net architecture.
    """
    return nn.Sequential(
        single_conv(in_channels, out_channels),
        single_conv(out_channels, out_channels)
    )

class UNet(nn.Module):
    """
    U-Net architecture for binary semantic segmentation.
    
    The network consists of an encoder (downsampling path) and decoder (upsampling path)
    with skip connections between corresponding encoder and decoder layers.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images)
        out_channels (int): Number of output channels (default: 1 for binary segmentation)
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.down1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = double_conv(512, 1024)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        # Encoder path with skip connections
        c1 = self.down1(x)      # 3 -> 64
        p1 = self.pool1(c1)
        c2 = self.down2(p1)     # 64 -> 128
        p2 = self.pool2(c2)
        c3 = self.down3(p2)     # 128 -> 256
        p3 = self.pool3(c3)
        c4 = self.down4(p3)     # 256 -> 512
        p4 = self.pool4(c4)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)  # 512 -> 1024
        
        # Decoder path with skip connections
        u4 = self.up4(bottleneck)         # 1024 -> 512
        u4 = torch.cat([u4, c4], dim=1)   # Skip connection: 512 + 512 -> 1024
        c5 = self.conv4(u4)               # 1024 -> 512
        
        u3 = self.up3(c5)                 # 512 -> 256
        u3 = torch.cat([u3, c3], dim=1)   # Skip connection: 256 + 256 -> 512
        c6 = self.conv3(u3)               # 512 -> 256
        
        u2 = self.up2(c6)                 # 256 -> 128
        u2 = torch.cat([u2, c2], dim=1)   # Skip connection: 128 + 128 -> 256
        c7 = self.conv2(u2)               # 256 -> 128
        
        u1 = self.up1(c7)                 # 128 -> 64
        u1 = torch.cat([u1, c1], dim=1)   # Skip connection: 64 + 64 -> 128
        c8 = self.conv1(u1)               # 128 -> 64
        
        # Final output
        out = self.final_conv(c8)         # 64 -> 1
        return out