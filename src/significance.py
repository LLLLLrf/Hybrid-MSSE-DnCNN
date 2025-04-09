import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file that stores the PSNR and SSIM of each image
# CSV Format：model_name, image_index, PSNR, SSIM
df = pd.read_csv("results/model_eval_results.csv")

df_psnr = df.pivot(index='image_index', columns='model_name', values='PSNR')
df_ssim = df.pivot(index='image_index', columns='model_name', values='SSIM')

print("PSNR DataFrame shape:", df_psnr.shape)
print("SSIM DataFrame shape:", df_ssim.shape)

# Compare the two models
model_A = "DnCNN"
model_B = "Hybrid-MSSE-DnCNN"

# Filter out rows with missing values (i.e., ensure each image has results from both models)
psnr_data = df_psnr[[model_A, model_B]].dropna()
ssim_data = df_ssim[[model_A, model_B]].dropna()

# Extract the measurements
psnr_A = psnr_data[model_A].values
psnr_B = psnr_data[model_B].values

ssim_A = ssim_data[model_A].values
ssim_B = ssim_data[model_B].values

# Normality test for PSNR differences
psnr_diff = psnr_A - psnr_B
shapiro_psnr = stats.shapiro(psnr_diff)
print(f"PSNR difference Shapiro test: W={shapiro_psnr.statistic:.3f}, p={shapiro_psnr.pvalue:.3f}")

if shapiro_psnr.pvalue > 0.05:
    # Normally distributed, use paired t-test
    t_stat_psnr, p_val_psnr = stats.ttest_rel(psnr_A, psnr_B)
    print(f"PSNR paired t-test: t = {t_stat_psnr:.3f}, p = {p_val_psnr:.3e}")
else:
    # Not normally distributed, use Wilcoxon test
    w_stat_psnr, p_val_psnr = stats.wilcoxon(psnr_A, psnr_B)
    print(f"PSNR Wilcoxon test: statistic = {w_stat_psnr:.3f}, p = {p_val_psnr:.3e}")

# Normality test for SSIM differences
ssim_diff = ssim_A - ssim_B
shapiro_ssim = stats.shapiro(ssim_diff)
print(f"SSIM difference Shapiro test: W={shapiro_ssim.statistic:.3f}, p={shapiro_ssim.pvalue:.3f}")

if shapiro_ssim.pvalue > 0.05:
    # Normally distributed, use paired t-test
    t_stat_ssim, p_val_ssim = stats.ttest_rel(ssim_A, ssim_B)
    print(f"SSIM paired t-test: t = {t_stat_ssim:.3f}, p = {p_val_ssim:.3e}")
else:
    # Not normally distributed, use Wilcoxon test
    w_stat_ssim, p_val_ssim = stats.wilcoxon(ssim_A, ssim_B)
    print(f"SSIM Wilcoxon test: statistic = {w_stat_ssim:.3f}, p = {p_val_ssim:.3e}")


# Draw boxplot of PSNR and SSIM
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.boxplot(data=psnr_data)
plt.title("PSNR distribution")
plt.ylabel("PSNR (dB)")

plt.subplot(1, 2, 2)
sns.boxplot(data=ssim_data)
plt.title("SSIM distribution")
plt.ylabel("SSIM")
plt.tight_layout()
plt.savefig("boxplot.png")
plt.show()

