import torch
import torch.nn as nn
import numpy as np
import cv2
class SingleScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SingleScaleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.conv(x))

# -----------------------
# 预处理模块：实现γ校正、融合以及双边滤波
# -----------------------
class PreProcessing(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.4):
        """
        gamma: γ校正参数
        alpha: 融合因子，控制原始图与增强图的比例
        """
        super(PreProcessing, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, x):
        # 将 tensor 从 [-1,1] 转换到 [0,1]
        x = (x + 1) / 2  
        x = torch.clamp(x, 1e-6, 1.0)  # 避免后续 np.power() 出现 NaN

        # 将 tensor 转为 CPU 上的 numpy 数组（不参与梯度传播）
        x_cpu = x.detach().cpu().numpy()
        out_list = []
        B = x_cpu.shape[0]
        for i in range(B):
            img = x_cpu[i, 0, :, :]  # 假设单通道
            # ① γ校正
            enhanced = np.power(img, self.gamma)
            # ② 融合
            blended = (1 - self.alpha) * img + self.alpha * enhanced
            # ③ 双边滤波（参数可根据需求调整）
            filtered = cv2.bilateralFilter(blended.astype(np.float32), d=5, sigmaColor=0.05, sigmaSpace=50)
            out_list.append(filtered[None, ...])
        # 重组 batch，转换为 tensor，并恢复原始设备和数据类型
        out_np = np.stack(out_list, axis=0)  # shape: (B, 1, H, W)
        out_tensor = torch.from_numpy(out_np).to(x.device).type_as(x)
        return out_tensor
    
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2, bias=False)
            for k in kernel_sizes
        ])
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 分别使用不同尺寸卷积，再在通道维度上拼接
        conv_outs = [conv(x) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        return self.relu(out)

class HybridDnCNN_NoSE(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64):
        super(HybridDnCNN_NoSE, self).__init__()
        self.initial = MultiScaleConvBlock(channels, features, kernel_sizes=[3, 5, 7])
        self.conv1x1 = nn.Conv2d(features * 3, features, kernel_size=1, bias=False)
        
        # 构建中间残差块，不使用 SE 模块
        layers = []
        for _ in range(num_of_layers):
            layers.append(MultiScaleConvBlock(features, features, kernel_sizes=[3, 5, 7]))
            layers.append(nn.Conv2d(features * 3, features, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            # 不添加 SEBlock
        self.body = nn.Sequential(*layers)
        self.final = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.initial(x)
        out = self.conv1x1(out)
        out = self.body(out)
        noise = self.final(out)
        return x - noise

class HybridDnCNN_NoSE_WithPreprocessing(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, gamma=1.5, alpha=0.4):
        super(HybridDnCNN_NoSE_WithPreprocessing, self).__init__()
        self.preprocess = PreProcessing(gamma, alpha)
        self.model = HybridDnCNN_NoSE(channels, num_of_layers, features)
        
    def forward(self, x):
        x_pre = self.preprocess(x)
        output = self.model(x_pre)
        return output


class HybridDnCNN_NoMultiScale(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64):
        super(HybridDnCNN_NoMultiScale, self).__init__()
        # 第一层单尺度卷积模块
        self.initial = SingleScaleConvBlock(channels, features, kernel_size=3)
        
        # 构建中间残差块（全部使用单尺度卷积）
        layers = []
        for _ in range(num_of_layers):
            layers.append(SingleScaleConvBlock(features, features, kernel_size=3))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)
        
        # 最后一层卷积，输出噪声估计
        self.final = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.initial(x)
        out = self.body(out)
        noise = self.final(out)
        return x - noise

class HybridDnCNN_NoMultiScale_WithPreprocessing(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, gamma=1.5, alpha=0.4):
        super(HybridDnCNN_NoMultiScale_WithPreprocessing, self).__init__()
        self.preprocess = PreProcessing(gamma, alpha)
        self.model = HybridDnCNN_NoMultiScale(channels, num_of_layers, features)
        
    def forward(self, x):
        x_pre = self.preprocess(x)
        output = self.model(x_pre)
        return output