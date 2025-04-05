import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 读取存储了每张图像 PSNR 和 SSIM 的 CSV 文件
# CSV 格式：model_name, image_index, PSNR, SSIM
df = pd.read_csv("results/model_eval_results.csv")

# 将数据透视为每一行代表一张图像，各模型的 PSNR 数据在不同列
df_psnr = df.pivot(index='image_index', columns='model_name', values='PSNR')
df_ssim = df.pivot(index='image_index', columns='model_name', values='SSIM')

# 检查透视后的数据
print("PSNR DataFrame shape:", df_psnr.shape)
print("SSIM DataFrame shape:", df_ssim.shape)

# 选择要比较的两个模型，例如 "DnCNN" 和 "FFDNet"
model_A = "DnCNN"
model_B = "Hybrid-MSSE-DnCNN"
# model_A_label="DnCNN"
# model_B_label="Hybrid-MSSE-DnCNN"

# 过滤掉有缺失值的行（即保证每张图像都有两个模型的结果）
psnr_data = df_psnr[[model_A, model_B]].dropna()
ssim_data = df_ssim[[model_A, model_B]].dropna()

# 提取各自的测量值
psnr_A = psnr_data[model_A].values
psnr_B = psnr_data[model_B].values

ssim_A = ssim_data[model_A].values
ssim_B = ssim_data[model_B].values

# 对 PSNR 差值进行正态性检验
psnr_diff = psnr_A - psnr_B
shapiro_psnr = stats.shapiro(psnr_diff)
print(f"PSNR 差值 Shapiro 检验: W={shapiro_psnr.statistic:.3f}, p={shapiro_psnr.pvalue:.3f}")

if shapiro_psnr.pvalue > 0.05:
    # 服从正态分布，使用配对 t 检验
    t_stat_psnr, p_val_psnr = stats.ttest_rel(psnr_A, psnr_B)
    print(f"PSNR 配对 t 检验: t = {t_stat_psnr:.3f}, p = {p_val_psnr:.3e}")
else:
    # 不服从正态分布，使用 Wilcoxon 检验
    w_stat_psnr, p_val_psnr = stats.wilcoxon(psnr_A, psnr_B)
    print(f"PSNR Wilcoxon 检验: statistic = {w_stat_psnr:.3f}, p = {p_val_psnr:.3e}")

# 对 SSIM 差值进行正态性检验
ssim_diff = ssim_A - ssim_B
shapiro_ssim = stats.shapiro(ssim_diff)
print(f"SSIM 差值 Shapiro 检验: W={shapiro_ssim.statistic:.3f}, p={shapiro_ssim.pvalue:.3f}")

if shapiro_ssim.pvalue > 0.05:
    # 服从正态分布，使用配对 t 检验
    t_stat_ssim, p_val_ssim = stats.ttest_rel(ssim_A, ssim_B)
    print(f"SSIM 配对 t 检验: t = {t_stat_ssim:.3f}, p = {p_val_ssim:.3e}")
else:
    # 不服从正态分布，使用 Wilcoxon 检验
    w_stat_ssim, p_val_ssim = stats.wilcoxon(ssim_A, ssim_B)
    print(f"SSIM Wilcoxon 检验: statistic = {w_stat_ssim:.3f}, p = {p_val_ssim:.3e}")

# 绘制箱线图进行直观展示
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

