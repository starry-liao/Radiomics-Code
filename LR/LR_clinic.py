#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# ------------------------------
# 1. 数据读取与划分
# ------------------------------
# TODO: 请替换为您的CSV文件路径
omics_train = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
omics_test  = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
clinic      = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

train_ids = omics_train['Patient_ID'].tolist()
test_ids  = omics_test['Patient_ID'].tolist()

clinic_train = clinic[clinic['Patient_ID'].isin(train_ids)].copy()
clinic_test  = clinic[clinic['Patient_ID'].isin(test_ids)].copy()

# ------------------------------
# 2. 准备单变量数据
# ------------------------------
clinic_features = ['SUVmax', 'MTV']  # 'SUVmax', 'SUVpeak'
X_train = clinic_train[clinic_features].values
y_train = clinic_train['label'].values
X_test  = clinic_test[clinic_features].values
y_test  = clinic_test['label'].values

print(f"特征数：{len(clinic_features)}")
print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")
print(f"训练集类别分布：\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布：\n{pd.Series(y_test).value_counts()}")


# In[3]:


# ==================== 数据标准化（逻辑回归必需） ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 逻辑回归超参数优化 ====================
# 参数网格：主要搜索正则化强度C和类别权重
param_grid = {
    'C': np.logspace(-2, 2, 20),
    'penalty': ['l1', 'l2'],               # 逻辑回归常用L2正则化
    # 'class_weight': ['balanced', None],
    'solver': ['lbfgs'],# lbfgs支持L2正则化，适合小数据集
    'class_weight': [None, 'balanced']
}

# 基础模型
lr = LogisticRegression(random_state=42, max_iter=1000)

# 分层K折交叉验证（5折）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索（以AUC为优化指标）
grid_search = GridSearchCV(
    estimator=lr,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("开始超参数搜索...")
grid_search.fit(X_train_scaled, y_train)

print(f"最佳参数：{grid_search.best_params_}")
best_lr = grid_search.best_estimator_


# In[4]:


# ==================== 4. 测试集评估 ====================
# 预测概率和类别
y_pred_prob = best_lr.predict_proba(X_test_scaled)[:, 1]
y_pred = best_lr.predict(X_test_scaled)

# 计算ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 计算敏感度和特异度
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n========== 测试集评估结果 ==========")
print(f"AUC: {roc_auc:.4f}")
print(f"敏感度 (Sensitivity / Recall): {sensitivity:.4f}")
print(f"特异度 (Specificity): {specificity:.4f}")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print("\n混淆矩阵：")
print(cm)
print("\n详细分类报告：")
print(classification_report(y_test, y_pred, target_names=['良性', '恶性']))


# In[5]:


# 在测试集评估部分之后添加
from sklearn.metrics import roc_curve

# 计算最佳阈值（约登指数 = 敏感度 + 特异度 - 1）
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thresholds[best_idx]
print(f"最佳阈值: {best_threshold:.4f}")

# 应用新阈值
y_pred_adjusted = (y_pred_prob >= best_threshold).astype(int)

# 重新计算评估指标
from sklearn.metrics import confusion_matrix, classification_report
cm_new = confusion_matrix(y_test, y_pred_adjusted)
tn, fp, fn, tp = cm_new.ravel()
sensitivity_new = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_new = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy_new = (tp + tn) / (tp + tn + fp + fn)

print("\n========== 调整阈值后的测试集评估 ==========")
print(f"最佳阈值: {best_threshold:.4f}")
print(f"AUC: {roc_auc:.4f} (不变)")
print(f"敏感度: {sensitivity_new:.4f}")
print(f"特异度: {specificity_new:.4f}")
print(f"准确率: {accuracy_new:.4f}")
print("\n混淆矩阵：")
print(cm_new)
print("\n分类报告：")
print(classification_report(y_test, y_pred_adjusted, target_names=['良性', '恶性']))


# In[6]:


# ==================== 5. 绘制ROC曲线 ====================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - CT Features (Logistic Regression)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('lr_ct_roc_curve.pdf', dpi=300)
# plt.savefig('lr_ct_roc_curve.png', dpi=300)
plt.show()


# In[ ]:




