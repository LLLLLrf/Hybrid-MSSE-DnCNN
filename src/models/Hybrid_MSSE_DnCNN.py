import torch
import torch.nn as nn
import numpy as np
import cv2

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

# -----------------------
# 模型结构部分
# -----------------------

# 多尺度卷积模块：采用不同卷积核尺寸（例如3, 5, 7），提取多尺度特征
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

# SE注意力模块：自适应地对通道特征加权
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

# 创新网络结构：HybridDnCNN
class HybridDnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64):
        super(HybridDnCNN, self).__init__()
        # 第一层采用多尺度卷积模块，输出3倍 features 通道（依赖于 kernel_sizes 长度）
        self.initial = MultiScaleConvBlock(channels, features, kernel_sizes=[3, 5, 7])
        # 1x1卷积降维，将拼接后通道数恢复为 features
        self.conv1x1 = nn.Conv2d(features * 3, features, kernel_size=1, bias=False)
        
        # 构建中间残差块
        layers = []
        for _ in range(num_of_layers):
            layers.append(MultiScaleConvBlock(features, features, kernel_sizes=[3, 5, 7]))
            layers.append(nn.Conv2d(features * 3, features, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            layers.append(SEBlock(features))
        self.body = nn.Sequential(*layers)
        
        # 最后一层卷积，输出噪声估计
        self.final = nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        # 初始多尺度特征提取与融合
        out = self.initial(x)
        out = self.conv1x1(out)
        # 残差块处理
        out = self.body(out)
        noise = self.final(out)
        # 残差学习：直接用输入减去估计噪声
        return x - noise

# -----------------------
# 整合预处理模块与 HybridDnCNN 模型
# -----------------------
class HybridDnCNNWithPreprocessing(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, gamma=1.5, alpha=0.4):
        """
        channels: 图像通道数（例如灰度图为1）
        num_of_layers: 残差块数
        features: 基础特征通道数
        gamma, alpha: 预处理参数，与之前一致
        """
        super(HybridDnCNNWithPreprocessing, self).__init__()
        self.preprocess = PreProcessing(gamma, alpha)
        self.model = HybridDnCNN(channels, num_of_layers, features)
        
    def forward(self, x):
        # 先预处理，再送入主网络
        x_pre = self.preprocess(x)
        output = self.model(x_pre)

        return output

# -----------------------
# 示例测试
# -----------------------
if __name__ == "__main__":
    # 创建包含预处理的模型
    model = HybridDnCNNWithPreprocessing(channels=1, num_of_layers=10, features=64, gamma=1.5, alpha=0.4)
    # 构造一个随机输入（假设像素值已经归一化到[0, 1]）
    input_tensor = torch.randn(1, 1, 256, 256)
    output = model(input_tensor)
    print("输出尺寸:", output.shape)
