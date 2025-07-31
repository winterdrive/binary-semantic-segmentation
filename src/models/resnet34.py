import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    """
    Create a convolution block with Conv2d, BatchNorm, and ReLU.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Convolution padding
        
    Returns:
        nn.Sequential: Sequential block with conv, batch norm, and ReLU
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet architecture.
    
    This implements the basic building block used in ResNet-18 and ResNet-34,
    which consists of two 3x3 convolutions with a skip connection.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = conv_block(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        
        # Second convolution layer (without ReLU)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection downsampling if needed
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x  # Save input for skip connection
        
        # First convolution
        out = self.conv1(x)
        
        # Second convolution
        out = self.conv2(out)
        
        # Apply downsampling to skip connection if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class ResNet34(nn.Module):
    """
    ResNet-34 architecture implementation.
    
    ResNet-34 is a 34-layer deep convolutional neural network that uses
    residual connections to enable training of very deep networks.
    
    Args:
        num_classes (int): Number of output classes (default: 1000)
        input_channels (int): Number of input channels (default: 3)
    """
    
    def __init__(self, num_classes=1000, input_channels=3):
        super(ResNet34, self).__init__()
        
        # Initial layers
        self.conv1 = conv_block(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Four residual block layers
        self.layer1 = self._make_layer(64, 64, num_blocks=3, stride=1)    # 3 blocks, 64 channels
        self.layer2 = self._make_layer(64, 128, num_blocks=4, stride=2)   # 4 blocks, 128 channels
        self.layer3 = self._make_layer(128, 256, num_blocks=6, stride=2)  # 6 blocks, 256 channels
        self.layer4 = self._make_layer(256, 512, num_blocks=3, stride=2)  # 3 blocks, 512 channels
        
        # Classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a layer of residual blocks.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            num_blocks (int): Number of blocks in this layer
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential layer of residual blocks
        """
        downsample = None
        
        # Create downsampling layer if stride != 1 or channels change
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        
        # First block (may include downsampling)
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks (stride=1, no downsampling)
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through ResNet-34.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        # Initial layers
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# Example usage and testing
if __name__ == "__main__":
    model = ResNet34(num_classes=1000)
    
    # Test forward pass
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
