#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt
import joblib 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


# In[38]:


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


# In[39]:


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
y_true = combined_test['label'].values     # total_test里没有label？

print(f"联合特征数：{len(total_features)}")
print(f"测试集样本数：{X_test.shape[0]}")
print(f"测试集类别分布：\n{pd.Series(y_true).value_counts()}")


# In[40]:


# ==================== 3. 预测概率 ====================
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]   # 取正类的概率
y_prob_rf = model_rf.predict_proba(X_test)[:, 1]
y_prob_svm = model_svm.predict_proba(X_test)[:, 1]


# In[49]:


# ==============================
# 3. 定义DCA函数
# ==============================
def calculate_net_benefit(y_true, y_prob, thresholds):
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        nb = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


def plot_dca(y_true, model_probs_dict):
    thresholds = np.linspace(0, 0.8, 100)  # 医学常用区间

    plt.figure(figsize=(10, 6))

    # ===== 模型曲线 =====
    for name, probs in model_probs_dict.items():
        nb = calculate_net_benefit(y_true, probs, thresholds)
        # 使曲线更平滑
        # nb_smooth = lowess(nb, thresholds, frac=0.1)
        # plt.plot(nb_smooth[:,0], nb_smooth[:,1])
        
        plt.plot(thresholds, nb, label=name)

    # ===== Treat All =====
    prevalence = np.mean(y_true)
    treat_all = [
        prevalence - (1 - prevalence) * (pt / (1 - pt))
        for pt in thresholds
    ]
    plt.plot(thresholds, treat_all, linestyle='--', label='Treat All')

    # ===== Treat None =====
    plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', label='Treat None')

    # ===== 美化 =====
    plt.xlabel("Threshold Probability", fontsize=13, fontweight='bold')
    plt.ylabel("Net Benefit", fontsize=13, fontweight='bold')
    plt.title("Decision Curve Analysis", fontsize=14, fontweight='bold', pad=12)
    plt.legend(loc='lower right', fontsize=10, edgecolor='black', facecolor='white', framealpha=0.9, borderpad=0.4)
    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.ylim(-0.2, 0.6)
    plt.xlim(0.1, 0.85)   # 临床常用范围
    # plt.margins(x=0)      # 去掉自动边距
    # plt.savefig(r"REPLACE_WITH_YOUR_FILE.png", dpi=300)

    plt.show()


# In[50]:


# ==============================
# 4. 构建模型字典并绘图
# ==============================
model_probs = {
    "LR Model": y_prob_lr,
    "RF Model": y_prob_rf,
    "SVM Model": y_prob_svm,
}

plot_dca(y_true, model_probs)


# In[ ]:




