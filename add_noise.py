import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio.v2 as imageio
import os

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255)

def add_rician_noise(image, noise_level=25):
    real = image + np.random.normal(0, noise_level, image.shape)
    imag = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.sqrt(real**2 + imag**2)
    return np.clip(noisy_image, 0, 255)

noise_level = 25
for folder in os.listdir('./s-data/'):
    for mri in os.listdir('./s-data/' + folder):
        for slice in os.listdir('./s-data/' + folder + '/' + mri):
            image = imageio.imread('./s-data/' + folder + '/' + mri + '/' + slice, mode='F')
            # gaussian_noisy_image = add_gaussian_noise(image, mean=0, std=25)
            rician_noisy_image = add_rician_noise(image, noise_level=noise_level)
            rician_noisy_image = rician_noisy_image.astype(np.uint8)  # 转换为 uint8 格式
            if not os.path.exists('./n-data{}/'.format(noise_level) + folder + '/' + mri):
                os.makedirs('./n-data{}/'.format(noise_level) + folder + '/' + mri)
                
            imageio.imwrite('./n-data{}/'.format(noise_level) + folder + '/' + mri + '/' + slice, rician_noisy_image)

# gaussian_noisy_image = add_gaussian_noise(image, mean=0, std=25)

# rician_noisy_image = add_rician_noise(image, noise_level=25)

