#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import joblib


# In[2]:


# ---------------------------- 全局绘图设置 ----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.unicode_minus': False,
    # 'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})


# In[3]:


# ==================== 1. 加载模型和标准化器 ====================
# TODO: 请替换为您的模型文件(.pkl)路径
model_ct = joblib.load( r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_pet = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_combined = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# ==================== 2. 读取测试集 ====================
# TODO: 请替换为您的CSV文件路径
ct_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")       # CT测试集
# TODO: 请替换为您的CSV文件路径
pet_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")     # PET测试集
# TODO: 请替换为您的CSV文件路径
combined_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# 假设标签列名为 'label'，CT和PET标签应一致
y_true = ct_test['label'].values

# 读取特征列表（与训练时一致）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    ct_features = [line.strip() for line in f.readlines()]
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    pet_features = [line.strip() for line in f.readlines()]
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    combined_features = [line.strip() for line in f.readlines()]

# 确保特征存在于数据中
ct_features = [f for f in ct_features if f in ct_test.columns]
pet_features = [f for f in pet_features if f in pet_test.columns]
combined_features = [f for f in combined_features if f in combined_test.columns]

# ==================== 3. 特征提取与标准化 ====================
# CT特征
X_ct = ct_test[ct_features].values

# PET特征
X_pet = pet_test[pet_features].values

# 组合特征
X_combined = combined_test[combined_features].values

print(f"CT特征数：{len(ct_features)}, PET特征数：{len(pet_features)}, 组合特征数：{len(combined_features)}")


# In[4]:


# ==================== 4. 预测概率 ====================
y_prob_ct = model_ct.predict_proba(X_ct)[:, 1]
y_prob_pet = model_pet.predict_proba(X_pet)[:, 1]
y_prob_comb = model_combined.predict_proba(X_combined)[:, 1]

# ==================== 5. 计算AUC及95%置信区间（DeLong方法） ====================

def delong_auc_ci(y_true, y_pred):
    """计算AUC及其95%置信区间（DeLong方法）"""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=float)
    
    pos = y_true == 1
    neg = y_true == 0
    n_pos = np.sum(pos)
    n_neg = np.sum(neg)
    
    pred_pos = y_pred[pos]
    pred_neg = y_pred[neg]
    
    V10 = np.array([(np.sum(p > pred_neg) + 0.5 * np.sum(p == pred_neg)) / n_neg for p in pred_pos])
    V01 = np.array([(np.sum(pred_pos > n) + 0.5 * np.sum(pred_pos == n)) / n_pos for n in pred_neg])
    
    auc_val = np.mean(V10)
    se = np.sqrt(np.var(V10, ddof=1) / n_pos + np.var(V01, ddof=1) / n_neg)
    
    z = norm.ppf(0.975)
    ci_lower = max(0, auc_val - z * se)
    ci_upper = min(1, auc_val + z * se)
    
    return auc_val, ci_lower, ci_upper

# ROC曲线
fpr_CT, tpr_CT, _ = roc_curve(y_true, y_prob_ct)
fpr_PET, tpr_PET, _ = roc_curve(y_true, y_prob_pet)
fpr_comb, tpr_comb, _ = roc_curve(y_true, y_prob_comb)

# AUC及95%CI
auc_CT, ci_CT_low, ci_CT_up = delong_auc_ci(y_true, y_prob_ct)
auc_PET, ci_PET_low, ci_PET_up = delong_auc_ci(y_true, y_prob_pet)
auc_comb, ci_comb_low, ci_comb_up = delong_auc_ci(y_true, y_prob_comb)

print(f"CT AUC: {auc_CT:.3f} (95% CI: {ci_CT_low:.3f}-{ci_CT_up:.3f})")
print(f"PET AUC: {auc_PET:.3f} (95% CI: {ci_PET_low:.3f}-{ci_PET_up:.3f})")
print(f"PET+CT AUC: {auc_comb:.3f} (95% CI: {ci_comb_low:.3f}-{ci_comb_up:.3f})")


# In[5]:


# roc叠加图
plt.figure(figsize=(6, 5))  # 论文常用宽高比

colors = ['#EE6677', '#4477AA', '#CCBB44', '#228833']   # Paul Tol Bright
line_styles = ['-', '--', '-.', ':']

plt.plot(fpr_CT, tpr_CT, label=f'CT (AUC = {auc_CT:.3f}, 95% CI: {ci_CT_low:.3f}-{ci_CT_up:.3f})')
plt.plot(fpr_PET, tpr_PET, label=f'PET (AUC = {auc_PET:.3f}, 95% CI: {ci_PET_low:.3f}-{ci_PET_up:.3f})')
plt.plot(fpr_comb, tpr_comb, label=f'PET-CT (AUC = {auc_comb:.3f}, 95% CI: {ci_comb_low:.3f}-{ci_comb_up:.3f})')

# 绘制随机猜测对角线
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.2, alpha=0.8, label='Random')

# ---------------------------- 4. 美化图表 ----------------------------
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=13, fontweight='bold')
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=13, fontweight='bold')
plt.title('ROC Curves Comparison (SVM)', fontsize=14, fontweight='bold', pad=12)
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', fontsize=10, edgecolor='black', facecolor='white', framealpha=0.9, borderpad=0.4)
plt.tight_layout()

# 保存高分辨率图片
# plt.savefig(r"REPLACE_WITH_YOUR_FILE.png", dpi=300, bbox_inches='tight')
plt.show()


# In[6]:


# # DeLong函数

# def compute_midrank(x):
#     sorted_idx = np.argsort(x)
#     sorted_x = x[sorted_idx]
#     n = len(x)
#     midranks = np.zeros(n)
    
#     i = 0
#     while i < n:
#         j = i
#         while j < n and sorted_x[j] == sorted_x[i]:
#             j += 1
#         midranks[i:j] = 0.5 * (i + j - 1)
#         i = j
    
#     out = np.empty(n)
#     out[sorted_idx] = midranks
#     return out


# def delong_test(y_true, pred1, pred2):
#     y_true = np.array(y_true)
    
#     pos = y_true == 1
#     neg = y_true == 0

#     x1 = pred1[pos]
#     y1 = pred1[neg]
#     x2 = pred2[pos]
#     y2 = pred2[neg]

#     m = len(x1)
#     n = len(y1)

#     V10_1 = compute_midrank(x1)
#     V01_1 = compute_midrank(y1)
#     auc1 = (V10_1.sum() / m - (m + 1) / 2) / n

#     V10_2 = compute_midrank(x2)
#     V01_2 = compute_midrank(y2)
#     auc2 = (V10_2.sum() / m - (m + 1) / 2) / n

#     var1 = np.var(V10_1) / m + np.var(V01_1) / n
#     var2 = np.var(V10_2) / m + np.var(V01_2) / n

#     cov = 0  # 简化协方差（小样本常用）

#     z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2 * cov)
#     p = 2 * (1 - stats.norm.cdf(abs(z)))

#     return p


# In[7]:


# 模型比较

# p_CT_vs_PET = delong_test(y_true, y_prob_ct, y_prob_pet)
# p_CT_vs_comb = delong_test(y_true, y_prob_ct, y_prob_comb)
# p_PET_vs_comb = delong_test(y_true, y_prob_pet, y_prob_comb)

# print(f"CT vs PET p = {p_CT_vs_PET:.4f}")
# print(f"CT vs PET+CT p = {p_CT_vs_comb:.4f}")
# print(f"PET vs PET+CT p = {p_PET_vs_comb:.4f}")


# In[ ]:




