import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dataset import MRIDenoisingDataset
from utils import batch_PSNR, batch_SSIM
from torchvision import transforms
import datetime
from models.DnCNN import DnCNN
from models.SE_model import HybridDnCNN
from models.Hybrid_MSSE_DnCNN import HybridDnCNNWithPreprocessing
from models.hybrid_FFDNet import HybridFFDNetWithPreprocessing
from models.FFDNet import FFDNet
from models.no_se_model import HybridDnCNN_NoMultiScale_WithPreprocessing, HybridDnCNN_NoSE_WithPreprocessing

# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device: {}".format(device))
if torch.cuda.is_available():
    print("gpu number: {}".format(torch.cuda.device_count()))

def get_model_attr(model, attr_name):
    if isinstance(model, nn.DataParallel):
        return getattr(model.module, attr_name)
    else:
        return getattr(model, attr_name)

        
# Configuration
class Opt:
    batchSize = 64
    num_of_layers = 17
    epochs = 50
    milestone = 30
    lr = 1e-4
    model_name="hybrid_MSSE_DnCNN"
    outf = f"./logs/experiment_{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    val_ratio = 0.2
    s_root = "s-data"
    n_root = "n-data25"

def main():
    opt = Opt()
    os.makedirs(opt.outf, exist_ok=True)
    
    # Loading dataset
    print("Loading dataset...")
    full_dataset = MRIDenoisingDataset(
        s_root=opt.s_root,
        n_root=opt.n_root,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    )
    
    # Split dataset
    torch.manual_seed(3407)
    train_size = int((1 - opt.val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    loader_train = DataLoader(train_set, batch_size=opt.batchSize, shuffle=True, num_workers=4)
    loader_val = DataLoader(val_set, batch_size=opt.batchSize, shuffle=False, num_workers=4)
    
    # Model initialization
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
    
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
    
    # Training loop
    writer = SummaryWriter(opt.outf)
    best_psnr = 0.0
    
    for epoch in range(opt.epochs):
        model.train()
        # Adjust learning rate
        if epoch >= opt.milestone:
            optimizer.param_groups[0]['lr'] = opt.lr / 10.0
        
        # Training phase
        for i, (noisy, clean) in enumerate(loader_train):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            pred_noise = model(noisy)  # Noise Prediction
            loss = criterion(pred_noise, (noisy - clean))
            loss.backward()
            optimizer.step()
            
            # Calculate PSNR and SSIM
            with torch.no_grad():
                denoised = torch.clamp(noisy - pred_noise, -1.0, 1.0)
                psnr = batch_PSNR(denoised, clean, data_range=2.0)
                ssim = batch_SSIM(denoised, clean, data_range=2.0)
                
            print(f"Epoch [{epoch+1}/{opt.epochs}] Batch [{i+1}/{len(loader_train)}] "
                f"Loss: {loss.item():.4f} PSNR: {psnr:.2f} SSIM: {ssim:.4f}")
            
            # TensorBoard logging
            writer.add_scalar('Loss/train', loss.item(), epoch*len(loader_train)+i)
            writer.add_scalar('PSNR/train', psnr, epoch*len(loader_train)+i)
            writer.add_scalar('SSIM/train', ssim, epoch*len(loader_train)+i)
        
        # Validation phase
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for noisy, clean in loader_val:
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                pred_noise = model(noisy)
                denoised = torch.clamp(noisy - pred_noise, -1.0, 1.0)
                val_psnr += batch_PSNR(denoised, clean, 2.0)
        
        val_psnr /= len(loader_val)
        writer.add_scalar('PSNR/val', val_psnr, epoch)
        print(f"Validation PSNR: {val_psnr:.2f}")
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), os.path.join(opt.outf, 'best_model.pth'))

    writer.close()

if __name__ == "__main__":
    main()