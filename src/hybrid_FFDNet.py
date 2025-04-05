import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 纯FFDNet模型部分（修改版，适用于hybrid结构）
# -----------------------
class FFDNet(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, scale_factor=2, noise_level=25/255):
        """
        channels: 图像通道数（例如灰度图为1，彩色图为3）
        num_of_layers: 网络层数
        features: 基础特征通道数
        scale_factor: 下采样尺度（默认2，即通过 pixel_unshuffle 降采样）
        noise_level: 默认噪声水平
        """
        super(FFDNet, self).__init__()
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        
        # 修改 in_channels：既然对噪声图也执行了 pixel_unshuffle，
        # 则噪声图通道数由1变为 scale_factor^2
        in_channels = channels * (scale_factor ** 2) + (scale_factor ** 2)
        
        self.input_conv = nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False)
        
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.middle = nn.Sequential(*layers)
        
        # 输出通道数恢复为 channels*(scale_factor^2)
        self.output_conv = nn.Conv2d(features, channels * (scale_factor ** 2), kernel_size=3, padding=1, bias=False)
    
    def forward(self, x, noise_level=None):
        """
        x: 输入图像，形状 (B, channels, H, W)
        noise_level: 噪声水平，如果为 None 则使用默认噪声水平
        """
        if noise_level is None:
            noise_level = self.noise_level
        
        B, C, H, W = x.size()
        # 对输入图像使用 pixel_unshuffle，下采样到 (B, C*(scale_factor^2), H/scale_factor, W/scale_factor)
        x_down = F.pixel_unshuffle(x, self.scale_factor)
        
        # 构造噪声水平图，原始形状为 (B, 1, H, W)
        noise_level_map = torch.full((B, 1, H, W), noise_level, device=x.device, dtype=x.dtype)
        # 对噪声水平图也执行 pixel_unshuffle，得到 (B, 1*(scale_factor^2), H/scale_factor, W/scale_factor)
        noise_level_down = F.pixel_unshuffle(noise_level_map, self.scale_factor)
        
        # 拼接降采样后的图像与噪声水平图，通道数为 channels*(scale_factor^2) + (scale_factor^2)
        input_cat = torch.cat([x_down, noise_level_down], dim=1)
        
        out = self.input_conv(input_cat)
        out = self.middle(out)
        out = self.output_conv(out)
        
        # 使用 pixel_shuffle 恢复到原图尺寸
        out = F.pixel_shuffle(out, self.scale_factor)
        
        # 残差学习：输出为去噪后的图像
        return x - out

# -----------------------
# 整合预处理模块与 FFDNet 的 Hybrid 模型
# -----------------------
class HybridFFDNetWithPreprocessing(nn.Module):
    def __init__(self, channels, num_of_layers=15, features=64, gamma=1.5, alpha=0.4,
                 scale_factor=2, noise_level=25/255):
        """
        channels: 图像通道数（例如灰度图为1）
        num_of_layers: 网络层数
        features: 基础特征通道数
        gamma, alpha: 预处理参数
        scale_factor: 下采样尺度
        noise_level: 默认噪声水平
        """
        super(HybridFFDNetWithPreprocessing, self).__init__()
        self.preprocess = PreProcessing(gamma, alpha)
        self.model = FFDNet(channels, num_of_layers, features, scale_factor, noise_level)
        
    def forward(self, x):
        # 先进行预处理，再送入FFDNet
        x_pre = self.preprocess(x)
        output = self.model(x_pre)
        return output

# -----------------------
# 示例测试
# -----------------------
if __name__ == "__main__":
    # 创建包含预处理的Hybrid FFDNet模型（例如针对单通道灰度图，scale_factor=2）
    model = HybridFFDNetWithPreprocessing(channels=1, num_of_layers=10, features=64,
                                          gamma=1.5, alpha=0.4, scale_factor=2, noise_level=25/255)
    # 构造一个随机输入（假设输入已经归一化到 [-1,1]）
    input_tensor = torch.randn(6, 1, 256, 256)
    output = model(input_tensor)
    print("输出尺寸:", output.shape)
