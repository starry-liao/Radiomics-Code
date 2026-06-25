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
model_lr = joblib.load( r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_rf = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# TODO: 请替换为您的模型文件(.pkl)路径
model_svm = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# ========================================
# 读取选中的组学特征名（每行一个特征名）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    radiomics_features = [line.strip() for line in f if line.strip()]
print(f"组学特征数量: {len(radiomics_features)}")

# 读取组学数据（已划分训练/测试集）
# TODO: 请替换为您的CSV文件路径
omics_test  = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# 读取全部临床数据（包含 特征 和 label）
# TODO: 请替换为您的CSV文件路径
clinic = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# ==============================
# 2. 数据对齐与特征合并
# ==============================
# 2.1 按 Patient_ID 提取临床数据的 特征和 label
clinic_subset = clinic[['Patient_ID', 'MTV', 'label']]  # 'SUVmax', 'SUVpeak'

# 2.2 为训练集和测试集合并组学特征与临床特征
radio_data = omics_test[['Patient_ID'] + radiomics_features].copy()
 # 合并临床数据
test_data = radio_data.merge(clinic_subset, on='Patient_ID', how='inner')

# ==============================
# 3. 准备特征矩阵与标签
# ==============================
clinic_features = ['MTV']  # 'SUVmax', 'SUVpeak'
feature_cols = radiomics_features + clinic_features  # 所有特征列（组学 + 临床）
X_test = test_data[feature_cols].values
y_true = test_data['label'].values

print(f"组合特征数：{len(feature_cols)}")
print(f"测试集样本数：{X_test.shape[0]}")
print(f"测试集类别分布：\n{pd.Series(y_true).value_counts()}")


# In[4]:


# ==================== 4. 预测概率 ====================
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
y_prob_svm = model_svm.predict_proba(X_test)[:, 1]

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
fpr_lr, tpr_lr, _ = roc_curve(y_true, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_true, y_prob_rf)
fpr_svm, tpr_svm, _ = roc_curve(y_true, y_prob_svm)

# AUC及95%CI
auc_lr, ci_lr_low, ci_lr_up = delong_auc_ci(y_true, y_prob_lr)
auc_rf, ci_rf_low, ci_rf_up = delong_auc_ci(y_true, y_prob_rf)
auc_svm, ci_svm_low, ci_svm_up = delong_auc_ci(y_true, y_prob_svm)

print(f"LR AUC: {auc_lr:.3f} (95% CI: {ci_lr_low:.3f}-{ci_lr_up:.3f})")
print(f"RF AUC: {auc_rf:.3f} (95% CI: {ci_rf_low:.3f}-{ci_rf_up:.3f})")
print(f"SVM AUC: {auc_svm:.3f} (95% CI: {ci_svm_low:.3f}-{ci_svm_up:.3f})")


# In[5]:


# roc叠加图
plt.figure(figsize=(6, 5))  # 论文常用宽高比

colors = ['#EE6677', '#4477AA', '#CCBB44', '#228833']   # Paul Tol Bright
line_styles = ['-', '--', '-.', ':']

plt.plot(fpr_lr, tpr_lr, label=f'LR (AUC = {auc_lr:.3f}, 95% CI: {ci_lr_low:.3f}-{ci_lr_up:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC = {auc_rf:.3f}, 95% CI: {ci_rf_low:.3f}-{ci_rf_up:.3f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {auc_svm:.3f}, 95% CI: {ci_svm_low:.3f}-{ci_svm_up:.3f})')

# 绘制随机猜测对角线
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.2, alpha=0.8, label='Random')

# ---------------------------- 4. 美化图表 ----------------------------
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=13, fontweight='bold')
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=13, fontweight='bold')
plt.title('ROC Curves Comparison (Combined Models)', fontsize=14, fontweight='bold', pad=12)
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




