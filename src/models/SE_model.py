import torch
import torch.nn as nn

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
        # 第一层采用多尺度卷积模块，输出3倍features通道（依赖于kernel_sizes长度）
        self.initial = MultiScaleConvBlock(channels, features, kernel_sizes=[3, 5, 7])
        # 1x1卷积降维，将拼接后通道数恢复为features
        self.conv1x1 = nn.Conv2d(features * 3, features, kernel_size=1, bias=False)
        
        # 构建中间残差块，每个块包含：
        # - 多尺度卷积模块（输出3*features通道）
        # - 1x1降维
        # - 批归一化与ReLU激活
        # - SE注意力模块
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

# 示例测试
if __name__ == "__main__":
    model = HybridDnCNN(channels=1, num_of_layers=10, features=64)
    input_tensor = torch.randn(1, 1, 256, 256)
    output = model(input_tensor)
    print("输出尺寸:", output.shape)
