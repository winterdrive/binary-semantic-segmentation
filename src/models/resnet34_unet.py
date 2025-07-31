import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet34 import BasicBlock, conv_block


def double_conv(in_channels, out_channels):
    """
    Double convolution block used in U-Net decoder.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Returns:
        nn.Sequential: Sequential block with two conv-bn-relu blocks
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class ResNet34_UNet(nn.Module):
    """
    U-Net with ResNet-34 backbone for semantic segmentation.
    
    This architecture combines the feature extraction power of ResNet-34
    with the precise localization capabilities of U-Net's decoder structure.
    
    Args:
        num_classes (int): Number of output classes (default: 1 for binary segmentation)
        input_channels (int): Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, num_classes=1, input_channels=3):
        super(ResNet34_UNet, self).__init__()
        
        # ResNet-34 Encoder (downsampling path)
        # Initial layers
        self.conv1 = conv_block(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet-34 layers with skip connections
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)    # Output: H/4, W/4
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)   # Output: H/8, W/8
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)  # Output: H/16, W/16
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)  # Output: H/32, W/32
        
        # U-Net Decoder (upsampling path)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(512, 256)  # 512 = 256 + 256 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(256, 128)  # 256 = 128 + 128 (skip connection)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(128, 64)   # 128 = 64 + 64 (skip connection)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)   # 128 = 64 + 64 (skip connection)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a layer of residual blocks (same as ResNet-34).
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential layer of residual blocks
        """
        downsample = None
        
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through ResNet34-UNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, H, W)
        """
        # Store original input size for final upsampling
        original_size = x.shape[-2:]
        
        # Encoder path (ResNet-34 backbone)
        x1 = self.conv1(x)      # 64 channels, H/2, W/2
        x_pool = self.maxpool(x1)  # 64 channels, H/4, W/4
        
        x2 = self.layer1(x_pool)   # 64 channels, H/4, W/4
        x3 = self.layer2(x2)       # 128 channels, H/8, W/8
        x4 = self.layer3(x3)       # 256 channels, H/16, W/16
        x5 = self.layer4(x4)       # 512 channels, H/32, W/32
        
        # Decoder path with skip connections
        # Upsampling 1: 512 -> 256 channels
        up1 = self.up1(x5)
        up1 = torch.cat([up1, x4], dim=1)  # Concatenate with skip connection
        up1 = self.up_conv1(up1)
        
        # Upsampling 2: 256 -> 128 channels
        up2 = self.up2(up1)
        up2 = torch.cat([up2, x3], dim=1)  # Concatenate with skip connection
        up2 = self.up_conv2(up2)
        
        # Upsampling 3: 128 -> 64 channels
        up3 = self.up3(up2)
        up3 = torch.cat([up3, x2], dim=1)  # Concatenate with skip connection
        up3 = self.up_conv3(up3)
        
        # Upsampling 4: 64 -> 64 channels
        up4 = self.up4(up3)
        up4 = torch.cat([up4, x1], dim=1)  # Concatenate with skip connection
        up4 = self.up_conv4(up4)
        
        # Final output layer
        output = self.final_conv(up4)
        
        # Ensure output matches input size
        if output.shape[-2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        return output


# Example usage and testing
if __name__ == "__main__":
    model = ResNet34_UNet(num_classes=1, input_channels=3)
    
    # Test forward pass
    input_tensor = torch.randn(2, 3, 256, 256)
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with different input size
    input_tensor2 = torch.randn(1, 3, 512, 512)
    output2 = model(input_tensor2)
    print(f"Input shape 2: {input_tensor2.shape}")
    print(f"Output shape 2: {output2.shape}")
