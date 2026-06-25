#!/usr/bin/env python
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import joblib 
import pandas as pd


# In[50]:


# ---------------------------- 全局绘图设置 ----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    # 'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})


# In[51]:


# ==================== 1. 加载模型 ====================
# TODO: 请替换为您的模型文件(.pkl)路径
model_lr = joblib.load( r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_rf = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_svm = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# ==================== 2. 读取测试数据 ====================
# 假设测试集为 CSV 文件，包含特征列和标签列 'label'
# TODO: 请替换为您的CSV文件路径
combined_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
clinic = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# 合并组学和临床数据
clinic_subset = clinic[['Patient_ID', 'MTV', 'label']]
total_test = combined_test.merge(clinic_subset, on='Patient_ID', how='inner')

# 读取特征列表
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    combined_features = [line.strip() for line in f.readlines()]

# 组学+临床特征
total_features = combined_features + ['MTV']

# 提取特征矩阵和标签
X_test = total_test[total_features].values
y_test = combined_test['label'].values     # total_test里没有label？

print(f"联合特征数：{len(total_features)}")
print(f"测试集样本数：{X_test.shape[0]}")
print(f"测试集类别分布：\n{pd.Series(y_test).value_counts()}")


# In[52]:


# ==================== 3. 预测概率 ====================
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]   # 取正类的概率
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
y_prob_svm = model_svm.predict_proba(X_test)[:, 1]

# ==================== 4. 计算校准曲线 ====================
# 将预测概率分成10个 bins（可调整 n_bins）
prob_true_lr, prob_pred_lr = calibration_curve(y_test, y_prob_lr, n_bins=5)
prob_true_rf, prob_pred_rf = calibration_curve(y_test, y_prob_rf, n_bins=5)
prob_true_svm, prob_pred_svm = calibration_curve(y_test, y_prob_svm, n_bins=5)

# 计算 Brier 分数
brier_lr = brier_score_loss(y_test, y_prob_lr)
brier_rf = brier_score_loss(y_test, y_prob_rf)
brier_svm = brier_score_loss(y_test, y_prob_svm)

print(f"LR Brier Score: {brier_lr:.4f}")
print(f"RF Brier Score: {brier_rf:.4f}")
print(f"SVM Brier Score: {brier_svm:.4f}")


# In[55]:


# ==================== 5. 绘制校准曲线 ====================
plt.figure(figsize=(8, 6))

# 绘制校准曲线
plt.plot(prob_pred_lr, prob_true_lr, marker='o', linewidth=2, label=f'LR (brier = {brier_lr:.3f})')
plt.plot(prob_pred_rf, prob_true_rf, marker='o', linewidth=2, label=f'RF (brier = {brier_rf:.3f})')
plt.plot(prob_pred_svm, prob_true_svm, marker='o', linewidth=2, label=f'SVM (brier = {brier_svm:.3f})')

# 绘制完美校准参考线 (y=x)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

plt.xlabel('Mean Predicted Probability (bin center)', fontsize=13, fontweight='bold')
plt.ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
plt.title('Calibration Curve', fontsize=14, fontweight='bold', pad=12)
plt.legend(loc='lower right', fontsize=10, edgecolor='black', facecolor='white', framealpha=0.9, borderpad=0.4)
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.tight_layout()
# plt.savefig('calibration_curve.pdf', dpi=300)
# plt.savefig(r"REPLACE_WITH_YOUR_FILE.png", dpi=300)
plt.show()


# In[54]:


# # ==================== 6. （可选）绘制带直方图的校准曲线 ====================
# # 改进版：在底部添加预测概率的直方图
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

# # 上子图：校准曲线
# ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Random Forest')
# ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
# ax1.set_xlabel('Mean Predicted Probability')
# ax1.set_ylabel('Fraction of Positives')
# ax1.set_title(f'Calibration Curve (Brier = {brier:.4f})')
# ax1.legend(loc='upper left')
# ax1.grid(alpha=0.3)

# # 下子图：预测概率的直方图
# ax2.hist(y_prob, bins=20, density=False, alpha=0.7, color='steelblue', edgecolor='black')
# ax2.set_xlabel('Predicted Probability')
# ax2.set_ylabel('Count')
# ax2.set_title('Distribution of Predicted Probabilities')
# ax2.grid(alpha=0.3)

# plt.tight_layout()
# # plt.savefig('calibration_curve_with_hist.pdf', dpi=300)
# # plt.savefig('calibration_curve_with_hist.png', dpi=300)
# plt.show()


# In[ ]:




