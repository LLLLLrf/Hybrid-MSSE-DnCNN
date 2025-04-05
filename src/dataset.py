import os
import random
import cv2
import torch
import torch.utils.data as udata
from torchvision import transforms

class MRIDenoisingDataset(udata.Dataset):
    def __init__(self, s_root, n_root, transform=None):
        super(MRIDenoisingDataset, self).__init__()
        self.pair_paths = []
        
        # 递归遍历所有子目录
        for root, dirs, files in os.walk(s_root):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # 仅处理图像文件
                    s_path = os.path.join(root, file)
                    
                    # 构建对应的噪声图像路径
                    relative_path = os.path.relpath(root, s_root)
                    n_root_dir = os.path.join(n_root, relative_path)
                    n_path = os.path.join(n_root_dir, file)
                    
                    # 检查噪声图像是否存在
                    if os.path.exists(n_path):
                        self.pair_paths.append((s_path, n_path))
                    else:
                        print(f"Warning: Noisy image not found: {n_path}")
        
        random.shuffle(self.pair_paths)
        self.transform = transform
        print(f"Total valid pairs: {len(self.pair_paths)}")

    def __len__(self):
        return len(self.pair_paths)

    def __getitem__(self, idx):
        s_path, n_path = self.pair_paths[idx]
                
        # 检查文件是否存在
        if not os.path.exists(s_path): #  判断s_path路径是否存在，如果不存在，则抛出文件未找到异常
            raise FileNotFoundError(f"Clean image not found: {s_path}")
        if not os.path.exists(n_path):
            raise FileNotFoundError(f"Noisy image not found: {n_path}")
        
        # 使用cv2读取图像
        s_img = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        n_img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        
        # 检查图像是否加载成功
        if s_img is None:
            raise ValueError(f"Failed to load clean image: {s_path}")
        if n_img is None:
            raise ValueError(f"Failed to load noisy image: {n_path}")
        
        # 调整图像尺寸为固定大小（例如256x256）
        target_size = (256, 256)
        s_img = cv2.resize(s_img, target_size)
        n_img = cv2.resize(n_img, target_size)
        
        # 转换为float32并归一化到[0,1]
        s_img = s_img.astype('float32') / 255.0
        n_img = n_img.astype('float32') / 255.0
        
        # 应用变换
        if self.transform:
            s_img = self.transform(s_img)
            n_img = self.transform(n_img)
            
        return n_img, s_img  # (noisy, clean)

    

# 转换定义
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
])
