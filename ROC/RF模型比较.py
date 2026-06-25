#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import joblib


# In[8]:


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


# In[9]:


# 加载模型
# TODO: 请替换为您的模型文件(.pkl)路径
model_CT = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_PET = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_combined = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# 加载测试集
# TODO: 请替换为您的CSV文件路径
test_CT = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
test_PET = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
test_combined = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# CT特征
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt") as f:
    ct_features = [line.strip() for line in f]

# PET特征
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt") as f:
    pet_features = [line.strip() for line in f]

# 融合特征（带前缀）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt") as f:
    combined_features = [line.strip() for line in f]
    

y_true = test_CT['label']  # 三个数据应一致

y_prob_CT = model_CT.predict_proba(test_CT[ct_features])[:, 1]
y_prob_PET = model_PET.predict_proba(test_PET[pet_features])[:, 1]
y_prob_combined = model_combined.predict_proba(test_combined[combined_features])[:, 1]

# ==================== 计算AUC及95%置信区间（DeLong方法） ====================

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
fpr_CT, tpr_CT, _ = roc_curve(y_true, y_prob_CT)
fpr_PET, tpr_PET, _ = roc_curve(y_true, y_prob_PET)
fpr_comb, tpr_comb, _ = roc_curve(y_true, y_prob_combined)

# AUC及95%CI
auc_CT, ci_CT_low, ci_CT_up = delong_auc_ci(y_true, y_prob_CT)
auc_PET, ci_PET_low, ci_PET_up = delong_auc_ci(y_true, y_prob_PET)
auc_comb, ci_comb_low, ci_comb_up = delong_auc_ci(y_true, y_prob_combined)

print(f"CT AUC: {auc_CT:.3f} (95% CI: {ci_CT_low:.3f}-{ci_CT_up:.3f})")
print(f"PET AUC: {auc_PET:.3f} (95% CI: {ci_PET_low:.3f}-{ci_PET_up:.3f})")
print(f"Combined AUC: {auc_comb:.3f} (95% CI: {ci_comb_low:.3f}-{ci_comb_up:.3f})")


# In[10]:


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
plt.title('ROC Curves Comparison (RF)', fontsize=14, fontweight='bold', pad=12)
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', fontsize=10, edgecolor='black', facecolor='white', framealpha=0.9, borderpad=0.4)
plt.tight_layout()

# 保存高分辨率图片
# plt.savefig(r"REPLACE_WITH_YOUR_FILE.png", dpi=300, bbox_inches='tight')
plt.show()


# In[11]:


# DeLong检验
# from scipy import stats

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
#         midrank = 0.5 * (i + j - 1)
#         midranks[i:j] = midrank
#         i = j
    
#     out = np.empty(n)
#     out[sorted_idx] = midranks
#     return out

# def delong_roc_test(y_true, pred1, pred2):
#     order = np.argsort(-pred1)
#     y_true = y_true.iloc[order].values
#     pred1 = pred1[order]
#     pred2 = pred2[order]

#     pos = y_true == 1
#     neg = y_true == 0

#     n1 = np.sum(pos)
#     n2 = np.sum(neg)

#     v10 = compute_midrank(pred1[pos])
#     v01 = compute_midrank(pred1[neg])
#     auc1 = (np.sum(v10) / n1 - (n1 + 1) / 2) / n2

#     v10 = compute_midrank(pred2[pos])
#     v01 = compute_midrank(pred2[neg])
#     auc2 = (np.sum(v10) / n1 - (n1 + 1) / 2) / n2

#     diff = auc1 - auc2
#     var = np.var(pred1 - pred2) / len(y_true)

#     z = diff / np.sqrt(var)
#     p = 2 * (1 - stats.norm.cdf(abs(z)))

#     return p


# In[12]:


# p_CT_vs_PET = delong_roc_test(y_true, y_prob_CT, y_prob_PET)
# p_CT_vs_comb = delong_roc_test(y_true, y_prob_CT, y_prob_combined)
# p_PET_vs_comb = delong_roc_test(y_true, y_prob_PET, y_prob_combined)

# print(f"CT vs PET p-value: {p_CT_vs_PET:.4f}")
# print(f"CT vs PET+CT p-value: {p_CT_vs_comb:.4f}")
# print(f"PET vs PET+CT p-value: {p_PET_vs_comb:.4f}")


# In[ ]:




