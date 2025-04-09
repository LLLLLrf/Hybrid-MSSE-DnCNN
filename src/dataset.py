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
        
        # Iterate over all files in the directory
        for root, dirs, files in os.walk(s_root):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    s_path = os.path.join(root, file)
                    
                    # Convert the relative path to the noisy image path
                    relative_path = os.path.relpath(root, s_root)
                    n_root_dir = os.path.join(n_root, relative_path)
                    n_path = os.path.join(n_root_dir, file)
                    
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
                
        # Check if the files exist
        if not os.path.exists(s_path):
            raise FileNotFoundError(f"Clean image not found: {s_path}")
        if not os.path.exists(n_path):
            raise FileNotFoundError(f"Noisy image not found: {n_path}")
        
        s_img = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        n_img = cv2.imread(n_path, cv2.IMREAD_GRAYSCALE)
        
        if s_img is None:
            raise ValueError(f"Failed to load clean image: {s_path}")
        if n_img is None:
            raise ValueError(f"Failed to load noisy image: {n_path}")
        
        # Adjust the size of the images to 256x256
        target_size = (256, 256)
        s_img = cv2.resize(s_img, target_size)
        n_img = cv2.resize(n_img, target_size)
        
        # Convert to float32 and normalize to [0,1]
        s_img = s_img.astype('float32') / 255.0
        n_img = n_img.astype('float32') / 255.0
        
        if self.transform:
            s_img = self.transform(s_img)
            n_img = self.transform(n_img)
            
        return n_img, s_img  # (noisy, clean)


# Convert the images to tensors and normalize them 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
