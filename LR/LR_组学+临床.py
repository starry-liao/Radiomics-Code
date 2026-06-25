#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, roc_auc_score
import joblib
import warnings
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')


# In[5]:


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

# 读取全部临床数据（包含 MTV 和 label）
# TODO: 请替换为您的CSV文件路径
clinic = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# ==============================
# 2. 数据对齐与特征合并
# ==============================
# 2.1 按 Patient_ID 提取临床数据的 MTV 和 label
clinic_subset = clinic[['Patient_ID', 'MTV', 'label']]  

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
clinic_features = ['MTV']  
feature_cols = radio_cols + clinic_features  # 所有特征列（组学 + 临床）
X_train = train_data[feature_cols].values
y_train = train_data['label'].values
X_test = test_data[feature_cols].values
y_test = test_data['label'].values

print(f"联合特征数量: {len(feature_cols)}")
print(f"训练集样本数: {len(train_data)}")
print(f"测试集样本数: {len(test_data)}")


# In[6]:


# ==================== 4. 构建pipeline ====================
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# ==============================
# 5. 训练优化逻辑回归模型（带交叉验证选 C）
# ==============================
# 参数网格：主要搜索正则化强度C和类别权重
param_grid = [
    # L2 正则（稳定主力）
    {
        'lr__penalty': ['l2'],
        'lr__C': [0.01, 0.1, 0.5, 1, 5, 10, 50],
        'lr__solver': ['lbfgs', 'liblinear'],
        'lr__class_weight': [None, 'balanced']
    },
    
    # L1 正则（特征选择能力）
    {
        'lr__penalty': ['l1'],
        'lr__C': [0.01, 0.1, 0.5, 1, 5, 10],
        'lr__solver': ['liblinear', 'saga'],
        'lr__class_weight': [None, 'balanced']
    }
]

# 分层K折交叉验证（5折）
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索（以AUC为优化指标）
grid_search = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("开始超参数搜索...")
grid_search.fit(X_train, y_train)

print(f"最佳参数：{grid_search.best_params_}")
best_lr_pipeline = grid_search.best_estimator_


# In[7]:


# ==============================
# 7. 在测试集上评估
y_pred_prob = best_lr_pipeline.predict_proba(X_test)[:, 1]
y_pred = best_lr_pipeline.predict(X_test)

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


# In[8]:


# 在测试集评估部分之后添加

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


# In[9]:


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


# In[10]:


# ==============================
# 9. （可选）输出特征系数，分析贡献
# ==============================
# 取出Pipeline中的Logistic模型
best_model = best_lr_pipeline.named_steps['lr']

# 提取系数
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': best_model.coef_[0]
})

# 按绝对值排序
coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

print(coef_df.head(6))


# In[11]:


import seaborn as sns
corr = train_data[feature_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# In[12]:


# ==========检验每个特征的独立AUC==========
for col in feature_cols:
    scaler2 = StandardScaler()
    X_test2 = scaler2.fit_transform(test_data[[col]])
    auc_single = roc_auc_score(y_test, X_test2)
    print(f"{col}: Test AUC = {auc_single:.3f}")


# In[14]:


# ==================== 保存模型和Pipeline ====================
# 直接保存整个 pipeline，它已经包含了训练好的标准化器
# joblib.dump(best_lr_pipeline,  r"REPLACE_WITH_YOUR_FILE.pkl")
# print("\n最佳模型 Pipeline 已保存为 best_lr_total_pipeline.pkl")


# In[ ]:




