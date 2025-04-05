# Hybrid-MSSE-DnCNN Model
A improved version of DnCNN for image denoising

## Model Structure
![Model Structure](./assets/Hybrid-MSSE-DnCNN.png)

### **Comparison of DnCNN and Hybrid-MSSE-DnCNN**
| Feature            | DnCNN | Hybrid-MSSE-DnCNN |
|--------------------|-------|------------------|
| **Preprocessing (Gamma, Blending, Bilateral Filter)** | ❌ No  | ✅ Yes |
| **Multi-Scale Convolution** | ❌ No  | ✅ Yes (3,5,7 kernels) |
| **SE Attention Mechanism** | ❌ No  | ✅ Yes (Channel-wise attention) |
| **1x1 Convolution for Reduction** | ❌ No  | ✅ Yes (After multi-scale feature extraction) |
| **Batch Normalization** | ✅ Yes | ✅ Yes |
| **Residual Learning** | ✅ Yes | ✅ Yes |


## Results
![Boxplot](./assets/boxplot.png)
![Results](./assets/Results.png)

## Installation
```bash
conda env create -f environment.yml
```

## Training
```bash
python ./src/train.py
```
