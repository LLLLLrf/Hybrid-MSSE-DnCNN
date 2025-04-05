import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

class MRIDenoisingDataset(Dataset):
    def __init__(self, s_root, n_root, transform=None):
        self.s_root = s_root
        self.n_root = n_root
        self.transform = transform
        self.pair_paths = []

        # 遍历 s-data 目录，获取所有 (3D 图像名称, 切片文件名) 组合
        for volume in os.listdir(s_root):
            volume_s_path = os.path.join(s_root, volume)
            volume_n_path = os.path.join(n_root, volume)
            
            if not os.path.isdir(volume_s_path):
                continue  # 跳过非目录项
            
            for img_name in os.listdir(volume_s_path):
                s_img_path = os.path.join(volume_s_path, img_name)
                n_img_path = os.path.join(volume_n_path, img_name)
                
                if os.path.exists(n_img_path):  # 确保噪声图像存在
                    self.pair_paths.append((s_img_path, n_img_path))
    
    def __len__(self):
        return len(self.pair_paths)
    
    def __getitem__(self, idx):
        s_img_path, n_img_path = self.pair_paths[idx]

        # 读取图片
        s_img = Image.open(s_img_path).convert("L")  # 假设是灰度图
        n_img = Image.open(n_img_path).convert("L")

        # 预处理
        if self.transform:
            s_img = self.transform(s_img)
            n_img = self.transform(n_img)

        return n_img, s_img  # 输入是噪声图，标签是原图

# 变换：转换为 Tensor 并归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为 [0,1] 范围的 Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
])

# 创建数据集
dataset = MRIDenoisingDataset(s_root="s-data", n_root="n-data", transform=transform)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# 测试数据加载
for noisy, clean in train_loader:
    print(noisy.shape, clean.shape)  # torch.Size([batch, 1, H, W])
    break
