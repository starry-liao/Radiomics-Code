#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix


# In[8]:


# ------------------------------
# 1. 读取数据
# ------------------------------
# TODO: 请替换为您的CSV文件路径
df = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
print("数据预览：")
print(df.head())

# 确保列名存在
assert 'SUVmax' in df.columns, "数据中缺少 MTV 列"
assert 'label' in df.columns, "数据中缺少 label 列"

# 提取预测变量与真实标签
y_true = df['label'].values      # 真实标签（0:良性，1:恶性）
y_score = df['SUVpeak'].values       # 预测分值（MTV 值）


# In[9]:


# ------------------------------
# 2. 计算 ROC 曲线及 AUC
# ------------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
print(f"\nAUC = {roc_auc:.4f}")

# ------------------------------
# 3. 寻找最佳截断值（基于约登指数）
# ------------------------------
# 约登指数 = TPR - FPR
youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]
best_specificity = 1 - best_fpr

print("\n最佳截断值（约登指数最大）：")
print(f"  阈值 = {best_threshold:.4f}")
print(f"  灵敏度 (TPR) = {best_tpr:.4f}")
print(f"  特异度 (1-FPR) = {best_specificity:.4f}")
print(f"  约登指数 = {youden_index[best_idx]:.4f}")


# In[10]:


# （可选）输出该阈值下的混淆矩阵
y_pred = (y_score >= best_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"\n混淆矩阵（阈值={best_threshold:.4f}）：")
print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")


# In[11]:


# ------------------------------
# 4. 绘制 ROC 曲线并标注最佳阈值点
# ------------------------------
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')

# 标记最佳截断点
plt.scatter(best_fpr, best_tpr, color='red', s=80,
            label=f'Best threshold = {best_threshold:.3f}')

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('1 - Specificity (False Positive Rate)')
plt.ylabel('Sensitivity (True Positive Rate)')
plt.title('ROC Curve for MTV in Malignancy Prediction')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

# 保存图片（可选）
# plt.savefig('MTV_ROC_curve.png', dpi=300)

plt.show()


# In[ ]:




