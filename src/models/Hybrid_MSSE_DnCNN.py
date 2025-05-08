import torch
import torch.nn as nn
import numpy as np
import cv2

# -----------------------
# Preprocess Module: Implement gamma correction, blending, and bilateral filtering
# -----------------------
class PreProcessing(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.4):
        """
        gamma: gamma correction parameter
        alpha: blending factor, controlling the ratio of the original image and the enhanced image
        """
        super(PreProcessing, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, x):
        # convert tensor from [-1,1] to [0,1]
        x = (x + 1) / 2  
        x = torch.clamp(x, 1e-6, 1.0)

        # Convert tensor to numpy array on CPU (not involved in gradient propagation)
        x_cpu = x.detach().cpu().numpy()
        out_list = []
        B = x_cpu.shape[0]
        for i in range(B):
            img = x_cpu[i, 0, :, :]
            # ① γ correction
            enhanced = np.power(img, self.gamma)
            # ② blending
            blended = (1 - self.alpha) * img + self.alpha * enhanced
            # ③ bilateral filtering
            filtered = cv2.bilateralFilter(blended.astype(np.float32), d=5, sigmaColor=0.05, sigmaSpace=50)
            out_list.append(filtered[None, ...])
        # reorganize batch, convert to tensor, and restore original device and data type
        out_np = np.stack(out_list, axis=0)  # shape: (B, 1, H, W)
        out_tensor = torch.from_numpy(out_np).to(x.device).type_as(x)
        return out_tensor

# -----------------------
# Model Structure
# -----------------------

# Multi-scale convolution module: extract multi-scale features using different convolution kernel sizes (e.g., 3, 5, 7)
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Use different convolution kernel sizes to extract multi-scale features, and concatenate them along the channel dimension
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        return self.relu(out)

# SE attention module: adaptively weight channel features
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# HybridDnCNN
class HybridDnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64):
        super(HybridDnCNN, self).__init__()
        # The first layer uses a multi-scale convolution module, and the output has 3 times the number of features channels (dependent on the length of kernel_sizes)
        self.initial = MultiScaleConvBlock(channels, features, kernel_sizes=[3, 5, 7])
        # x1 convolution to reduce the number of channels back to features
        self.conv1x1 = nn.Conv2d(features * 3, features, kernel_size=1, bias=False)
        
        # Build residual blocks
        layers = []
        for _ in range(num_of_layers):
            layers.append(MultiScaleConvBlock(features, features, kernel_sizes=[3, 5, 7]))
            layers.append(nn.Conv2d(features * 3, features, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SEBlock(features))
        self.body = nn.Sequential(*layers)
        
        # The final convolution layer, outputting the estimated noise
        self.final = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        # Initial multi-scale feature extraction and fusion
        out = self.initial(x)
        out = self.conv1x1(out)
        # Residual block processing
        out = self.body(out)
        noise = self.final(out)
        # Residual learning: directly subtract the estimated noise from the input
        return x - noise

# -----------------------
# Integrate the preprocessing module with the HybridDnCNN model
# -----------------------
class HybridDnCNNWithPreprocessing(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, gamma=1.5, alpha=0.4):
        """
        channels: Number of image channels (e.g., 1 for grayscale images)
        num_of_layers: Number of residual blocks
        features: Basic feature channel number
        gamma, alpha: Preprocessing parameters, consistent with the previous version
        """
        super(HybridDnCNNWithPreprocessing, self).__init__()
        self.preprocess = PreProcessing(gamma, alpha)
        self.model = HybridDnCNN(channels, num_of_layers, features)
        
    def forward(self, x):
        # Preprocess the input tensor, and then pass it to the main network
        x_pre = self.preprocess(x)
        output = self.model(x_pre)

        return output

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    model = HybridDnCNNWithPreprocessing(channels=1, num_of_layers=10, features=64, gamma=1.5, alpha=0.4)

    input_tensor = torch.randn(1, 1, 256, 256)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    print("model:", model)