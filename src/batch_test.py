import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
from dataset import MRIDenoisingDataset
from utils import batch_PSNR, batch_SSIM
import os
from models.DnCNN import DnCNN
from models.SE_model import HybridDnCNN
from models.Hybrid_MSSE_DnCNN import HybridDnCNNWithPreprocessing
from models.hybrid_FFDNet import HybridFFDNetWithPreprocessing
from models.FFDNet import FFDNet
from models.no_se_model import HybridDnCNN_NoMultiScale_WithPreprocessing, HybridDnCNN_NoSE_WithPreprocessing

import random
import csv
from pytorch_msssim import ssim as calc_ssim
import torch.nn.functional as F
from math import log10

def compute_psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * log10(data_range) - 10 * torch.log10(mse).item()

def compute_ssim(img1, img2, data_range=1.0):
    # img shape: (1, H, W), must be 4D tensor
    return calc_ssim(img1.unsqueeze(0), img2.unsqueeze(0), data_range=data_range).item()


# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_list = ["DnCNN", "bilateral", "bilateral", "se", "FFDNet", "hybrid_FFNet"]
ckps = [
    "logs/experiment_DnCNN_20250313-013010/best_model.pth",
    "logs/experiment_bilateral_20250317-034747/best_model.pth",
    "logs/experiment_bilateral_20250321-010647/best_model.pth",
    "logs/experiment_se_20250314-025447/best_model.pth",
    "logs/experiment_FFDNet_20250319-023616/best_model.pth",
    "logs/experiment_hybrid_FFNet_20250319-113209/best_model.pth"
]

# 配置参数
class Opt:
    batchSize = 8
    val_ratio = 0.2
    num_of_layers = 17
    model_name = "se"
    checkpoint = "logs/experiment_se_20250314-025447/best_model.pth"

save = False
sample_num = 500

def main():
    opt = Opt()
    
    # 加载数据集
    full_dataset = MRIDenoisingDataset(
        s_root="s-data",
        n_root="n-data25",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    torch.manual_seed(3407)
    train_size = int((1 - opt.val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_set = random_split(full_dataset, [train_size, val_size])
    
    # 抽样 500 个索引
    sampled_indices = random.sample(range(len(val_set)), min(sample_num, len(val_set)))
    sampled_subset = torch.utils.data.Subset(val_set, sampled_indices)
    loader_val = DataLoader(sampled_subset, batch_size=opt.batchSize, shuffle=False, num_workers=4)

    # 保存 CSV 的目录
    os.makedirs("results", exist_ok=True)
    result_csv = open("results/model_eval_results.csv", "w", newline="")
    csv_writer = csv.writer(result_csv)
    csv_writer.writerow(["model_name", "image_index", "PSNR", "SSIM"])

    def adapt_state_dict(model, loaded_state_dict):
        new_state_dict = {}
        for k, v in loaded_state_dict.items():
            new_k = k.replace("module.", "")
            new_state_dict[new_k] = v

        model_keys = list(model.state_dict().keys())
        loaded_keys = list(new_state_dict.keys())
        prefixes = set(k.split('.')[0] for k in model_keys if '.' in k)
        if len(prefixes) == 1:
            prefix = list(prefixes)[0]
            if not any(k.startswith(prefix + ".") for k in loaded_keys):
                adapted_state_dict = {}
                for k, v in new_state_dict.items():
                    adapted_state_dict[prefix + "." + k] = v
                return adapted_state_dict
        return new_state_dict

    for ind, model_name in enumerate(model_list):
        opt.checkpoint = ckps[ind]
        opt.model_name = model_name

        if opt.model_name == "DnCNN":
            model = DnCNN(channels=1, num_of_layers=opt.num_of_layers).to(device)
        elif opt.model_name == "se":
            model = HybridDnCNN(channels=1, num_of_layers=opt.num_of_layers).to(device)
        elif opt.model_name == "hybrid_MSSE_DnCNN":
            model = HybridDnCNNWithPreprocessing(channels=1, num_of_layers=opt.num_of_layers, gamma=1.5, alpha=0.4).to(device)
        elif opt.model_name == "hybrid_FFNet":
            model = HybridFFDNetWithPreprocessing(channels=1, num_of_layers=opt.num_of_layers, gamma=1.5, alpha=0.4,
                                                scale_factor=2, noise_level=25/255).to(device)
        elif opt.model_name == "FFDNet":
            model = FFDNet(channels=1, num_of_layers=opt.num_of_layers, features=64, scale_factor=2, noise_level=25/255).to(device)
        elif opt.model_name == "no_se":
            model = HybridDnCNN_NoSE_WithPreprocessing(channels=1, num_of_layers=opt.num_of_layers, gamma=1.5, alpha=0.4).to(device)
        elif opt.model_name == "no_ms":
            model = HybridDnCNN_NoMultiScale_WithPreprocessing(channels=1, num_of_layers=opt.num_of_layers, gamma=1.5, alpha=0.4).to(device)
        else:
            raise ValueError("Invalid model name")

        state_dict = torch.load(opt.checkpoint, map_location=device)
        adapted_state_dict = adapt_state_dict(model, state_dict)
        model.load_state_dict(adapted_state_dict)
        model.eval()

        image_index = 0
        with torch.no_grad():
            for noisy, clean in loader_val:
                noisy = noisy.to(device)
                clean = clean.to(device)
                pred_noise = model(noisy)
                denoised = torch.clamp(noisy - pred_noise, -1.0, 1.0)

                for i in range(noisy.size(0)):
                    psnr = compute_psnr(denoised[i], clean[i], data_range=2.0)
                    ssim = compute_ssim(denoised[i], clean[i], data_range=2.0)
                    csv_writer.writerow([model_name, image_index, psnr, ssim])
                    image_index += 1

                if save and image_index < 8:
                    os.makedirs(f"results/{model_name}", exist_ok=True)
                    def unnormalize(img, mean=0.5, std=0.5):
                        return img * std + mean
                    for i in range(opt.batchSize):
                        torchvision.utils.save_image(unnormalize(denoised[i]), f"results/{model_name}/denoised_{i}.png")
                        torchvision.utils.save_image(unnormalize(pred_noise[i]), f"results/{model_name}/pred_noise_{i}.png")
                        torchvision.utils.save_image(unnormalize(clean[i]), f"results/{model_name}/clean_{i}.png")
                        torchvision.utils.save_image(unnormalize(noisy[i]), f"results/{model_name}/noisy_{i}.png")


        print(f"[{model_name}] finished. {image_index} samples evaluated.")

    result_csv.close()

if __name__ == "__main__":
    main()
    