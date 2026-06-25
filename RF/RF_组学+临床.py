#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# ==============================
# 1. 读取数据与特征列表
# ==============================
# 读取选中的组学特征名（每行一个特征名）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    radiomics_features = [line.strip() for line in f if line.strip()]
print(f"组学特征数量: {len(radiomics_features)}")

# 读取组学数据（已划分训练/测试集）
# TODO: 请替换为您的CSV文件路径
omics_train = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
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
def merge_features(omics_df, clinic_subset):
    # 提取组学特征列（只保留选中的特征）
    available_radio = [f for f in radiomics_features if f in omics_df.columns]
    if len(available_radio) < len(radiomics_features):
        print(f"警告: 有 {len(radiomics_features)-len(available_radio)} 个特征在数据中不存在")
    radio_data = omics_df[['Patient_ID'] + available_radio].copy()
    # 合并临床数据
    merged = radio_data.merge(clinic_subset, on='Patient_ID', how='inner')
    return merged, available_radio

train_data, radio_cols = merge_features(omics_train, clinic_subset)
test_data, _ = merge_features(omics_test, clinic_subset)

# ==============================
# 3. 准备特征矩阵与标签
# ==============================
feature_cols = radio_cols + ['MTV']  # 所有特征列（组学 + 临床）
X_train = train_data[feature_cols].values
y_train = train_data['label'].values
X_test = test_data[feature_cols].values
y_test = test_data['label'].values

print(f"组合特征数：{len(feature_cols)}")
print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")
print(f"训练集类别分布：\n{pd.Series(y_train).value_counts()}")
print(f"测试集类别分布：\n{pd.Series(y_test).value_counts()}")


# In[3]:


# ==================== 2. 随机森林超参数优化 ====================
# 参数网格（可根据数据量适当调整）
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [10, 20, 30],
    'min_samples_leaf': [5,10,15],
    'class_weight': ['balanced', None],
    'max_features': [0.3, 0.5, 'sqrt', 'log2', None],
    # 'criterion': ['gini', 'entropy']
}

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7],
#     'min_samples_split': [5, 10, 20],
#     'min_samples_leaf': [2, 5, 10],
#     'max_features': ['sqrt', 'log2', 0.5],
#     'class_weight': ['balanced']
# }

# 基础模型
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# 分层K折交叉验证（5折）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索（以AUC为优化指标）
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
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
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
plt.title('ROC Curve - PET+CT Combined Features (Random Forest)', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.savefig('rf_combined_roc_curve.pdf', dpi=300)
# plt.savefig('rf_combined_roc_curve.png', dpi=300)
plt.show()


# In[7]:


# ==================== 5. 特征重要性（可选） ====================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest - PET+CT)", fontsize=14)
plt.bar(range(len(feature_cols)), importances[indices], align="center")
plt.xticks(range(len(feature_cols)), np.array(feature_cols)[indices], rotation=90)
plt.tight_layout()
# plt.savefig('rf_combined_feature_importance.pdf', dpi=300)
plt.show()


# In[8]:


print("\n特征重要性排序（前10）：")
for i in range(min(10, len(feature_cols))):
    print(f"{feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")


# In[10]:


# # ==================== 6. 保存模型 ====================
# joblib.dump(best_rf, r"REPLACE_WITH_YOUR_FILE.pkl")
# print("\n最佳模型已保存为 best_rf_total_model.pkl")


# In[ ]:




