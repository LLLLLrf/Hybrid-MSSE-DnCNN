import nibabel as nib
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

dir = ["data/IXI-DTI","data/IXI-MRA","data/IXI-PD","data/IXI-T1","data/IXI-T2"]

def read_nii(nii_path):
    try:
        nii = nib.load(nii_path)
        data = nii.get_fdata()
    except Exception as e:
        print(f"Error reading {nii_path}: {e}")
        return None
    return data

def save_slices(data, save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i in range(data.shape[2]):
        plt.imsave(os.path.join(save_dir, f"slice_{i}.png"), data[:, :, i], cmap="gray")
    

for i in dir:
    print(i)
    for file in os.listdir(i):
        if file.endswith(".nii.gz"):
            data = read_nii(os.path.join(i, file))
            if data is not None:
                save_slices(data, os.path.join("s2-" + i, file.split(".")[0]))

