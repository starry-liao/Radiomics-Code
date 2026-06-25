#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
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
# 2. 准备变量数据
# ------------------------------
clinic_features = ['MTV']  # 'SUVmax', 'SUVpeak'
X_train = clinic_train[clinic_features].values
y_train = clinic_train['label'].values
X_test  = clinic_test[clinic_features].values
y_test  = clinic_test['label'].values

print(f"特征数：{len(clinic_features)}")
print(f"训练集样本数：{X_train.shape[0]}, 特征数：{X_train.shape[1]}")
print(f"测试集样本数：{X_test.shape[0]}, 特征数：{X_test.shape[1]}")
print(f"训练集类别分布：\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布：\n{pd.Series(y_test).value_counts()}")


# In[3]:


# ==================== 2. 随机森林超参数优化 ====================
# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy']
}

# 基础模型
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# 分层K折交叉验证（5折）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索（以AUC为优化目标）
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
best_rf = grid_search.best_estimator_


# In[4]:


# ==================== 3. 测试集评估 ====================
# 预测概率和类别
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]  # 正类概率
y_pred = best_rf.predict(X_test)

# 计算ROC曲线和AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# 计算敏感度（Sensitivity/Recall）和特异度（Specificity）
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


# ==================== 4. 绘制ROC曲线 ====================
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - CT Features + Random Forest', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('rf_ct_roc_curve.pdf', dpi=300)
# plt.savefig('rf_ct_roc_curve.png', dpi=300)
plt.show()


# In[7]:


# ==================== 5. 特征重要性（可选） ====================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest - CT)", fontsize=14)
plt.bar(range(len(clinic_features)), importances[indices], align="center")
plt.xticks(range(len(clinic_features)), np.array(clinic_features)[indices], rotation=90)
plt.tight_layout()
# plt.savefig('rf_ct_feature_importance.pdf', dpi=300)
plt.show()

print("\n特征重要性排序：")
for i in range(len(clinic_features)):
    print(f"{clinic_features[indices[i]]}: {importances[indices[i]]:.4f}")


# In[8]:


# ==================== 6. 保存模型（可选） ====================
# import joblib
# joblib.dump(best_rf, 'best_rf_ct_model.pkl')
# print("\n最佳模型已保存为 best_rf_ct_model.pkl")


# In[ ]:




