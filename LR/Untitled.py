#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# In[ ]:


# ---------------------------- 全局绘图设置 ----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'axes.unicode_minus': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})


# In[ ]:


# ---------------------------- 1. 加载模型 ----------------------------
model_paths = [
# TODO: 请替换为您的模型文件(.pkl)路径
    r"REPLACE_WITH_YOUR_FILE.pkl",
# TODO: 请替换为您的模型文件(.pkl)路径
    r"REPLACE_WITH_YOUR_FILE.pkl",
# TODO: 请替换为您的模型文件(.pkl)路径
    r"REPLACE_WITH_YOUR_FILE.pkl",
# TODO: 请替换为您的模型文件(.pkl)路径
    r"REPLACE_WITH_YOUR_FILE.pkl",
]
model_labels = ['CT', 'PET', 'PET-CT', 'Total']  # 用于图例的模型名称

pipelines = []
for path in model_paths:
    pipelines.append(joblib.load(path))

# ---------------------------- 2. 加载测试集 ----------------------------
# ==================== 2. 读取测试集 ====================
# TODO: 请替换为您的CSV文件路径
ct_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")       # CT测试集
# TODO: 请替换为您的CSV文件路径
pet_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")     # PET测试集
# TODO: 请替换为您的CSV文件路径
combined_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# 假设标签列名为 'label'，CT和PET标签应一致
y_true = ct_test['label'].values


# In[ ]:


# ---------------------------- 3. 计算 ROC 曲线 ----------------------------
plt.figure(figsize=(6, 5.5))  # 论文常用宽高比

colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']  # 可通过 colorblind 友好配色
line_styles = ['-', '--', '-.', ':']

for idx, (pipe, label) in enumerate(zip(pipelines, model_labels)):
    # 获取预测概率（正类概率，即 [:, 1]）
    y_score = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr,
             color=colors[idx % len(colors)],
             linestyle=line_styles[idx % len(line_styles)],
             linewidth=1.8,
             label=f'{label} (AUC = {roc_auc:.3f})')

# 绘制随机猜测对角线
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.2, alpha=0.8, label='Random')

# ---------------------------- 4. 美化图表 ----------------------------
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('1 - Specificity (False Positive Rate)', fontsize=13, fontweight='bold')
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=13, fontweight='bold')
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold', pad=12)
plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', fontsize=10, edgecolor='black',
           facecolor='white', framealpha=0.9, borderpad=0.4)
plt.tight_layout()

# 保存高分辨率图片
# plt.savefig('roc_comparison.png', dpi=600, bbox_inches='tight')
plt.show()

