import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # 修改 in_channels：既然对噪声图也进行了 pixel_unshuffle，
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
        x: 输入图像，形状 (B, channels, H, W)，数值范围应归一化到 [0, 1] 或 [-1, 1]
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
        
        # 拼接降采样后的图像与噪声水平图，通道数为 channels*(scale_factor^2) + scale_factor^2
        input_cat = torch.cat([x_down, noise_level_down], dim=1)
        
        out = self.input_conv(input_cat)
        out = self.middle(out)
        out = self.output_conv(out)
        
        # 使用 pixel_shuffle 恢复到原图尺寸
        out = F.pixel_shuffle(out, self.scale_factor)
        
        # 残差学习：输出为去噪后的图像
        return x - out

# -----------------------
# 示例测试
# -----------------------
if __name__ == "__main__":
    # 例如针对单通道灰度图，scale_factor=2
    model = FFDNet(channels=1, num_of_layers=10, features=64, scale_factor=2, noise_level=25/255)
    # 构造一个随机输入（假设输入已经归一化到 [0, 1] 或 [-1, 1]）
    input_tensor = torch.randn(6, 1, 256, 256)
    output = model(input_tensor)
    print("输出尺寸:", output.shape)
