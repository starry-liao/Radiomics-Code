#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# ==================== 1. 读取数据 ====================
# 读取训练集和测试集CSV文件（请修改为实际文件路径）
# TODO: 请替换为您的CSV文件路径
train_df = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")   # 包含所有特征和标签
# TODO: 请替换为您的CSV文件路径
test_df = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")     # 包含所有特征和标签

# 读取CT特征列表（每行一个特征名）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    combined_features = [line.strip() for line in f.readlines()]

# 确保特征在数据中存在
existing_features = [f for f in combined_features if f in train_df.columns]
missing = set(combined_features) - set(existing_features)
if missing:
    print(f"警告：以下特征在数据中不存在，已忽略：{missing}")

# 提取特征矩阵和标签（假设标签列名为 'label'）
X_train = train_df[existing_features].values
y_train = train_df['label'].values
X_test = test_df[existing_features].values
y_test = test_df['label'].values

print(f"特征数：{len(existing_features)}")
print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")
print(f"训练集类别分布：\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布：\n{pd.Series(y_test).value_counts()}")


# In[3]:


# ==================== 2. 使用 Pipeline 定义模型和标准化 ====================
# 创建一个 Pipeline，确保标准化在交叉验证的每一折中都正确进行
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42, probability=True))
])

# ==================== 3. 构建完善的参数网格 ====================
# 注意：参数名需要用 'svm__' 作为前缀，以对应 Pipeline 中的 'svm' 步骤
param_grid = {          #0.8515
    'svm__kernel': ['rbf'],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.1],
    'svm__class_weight': ['balanced']
}

# param_grid = {            #0.8655
#     'svm__kernel': ['linear'],
#     'svm__C': [0.01, 0.1, 0.5, 1, 5, 10, 50],
#     'svm__class_weight': [None, 'balanced']
# }

# param_grid = [
#     # RBF 核
#     {
#         'svm__C': [0.1, 1, 10, 100],
#         'svm__kernel': ['rbf'],
#         'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],
#         'svm__class_weight': [None, 'balanced']
#     },
#     线性核      0.8711
#     {
#         'svm__C': [0.1, 1, 10, 100],
#         'svm__kernel': ['linear'],
#         'svm__class_weight': [None, 'balanced']
#     },
#     # 多项式核
#     {
#         'svm__C': [0.1, 1, 10],
#         'svm__kernel': ['poly'],
#         'svm__degree': [2, 3],
#         'svm__gamma': ['scale', 'auto', 0.1],
#         'svm__coef0': [0, 1],
#         'svm__class_weight': [None, 'balanced']
#     },
#     # Sigmoid 核
#     {
#         'svm__C': [0.1, 1, 10],
#         'svm__kernel': ['sigmoid'],
#         'svm__gamma': ['scale', 'auto', 0.1],
#         'svm__coef0': [0, 1],
#         'svm__class_weight': [None, 'balanced']
#     },
# ]

# ==================== 4. SVM超参数优化 ====================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
best_svm_pipeline = grid_search.best_estimator_


# In[4]:


# ==================== 5. 测试集评估 ====================
y_pred_prob = best_svm_pipeline.predict_proba(X_test)[:, 1]
y_pred = best_svm_pipeline.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
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


# 约登指数平衡敏感度和特异度
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
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - CT Features (SVM)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('svm_ct_roc_curve.pdf', dpi=300)
# plt.savefig('svm_ct_roc_curve.png', dpi=300)
plt.show()


# In[8]:


# ==================== 7. 保存模型和Pipeline ====================
# 直接保存整个 pipeline，它已经包含了训练好的标准化器
# joblib.dump(best_svm_pipeline,  r"REPLACE_WITH_YOUR_FILE.pkl")
# print("\n最佳模型 Pipeline 已保存为 best_svm_combined_pipeline.pkl")


# In[ ]:




