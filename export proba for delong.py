#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[46]:


# # ==================== 1. 加载模型和标准化器 ====================
# model_ct = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# scaler_ct = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# model_pet = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# scaler_pet = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# model_combined = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")
# scaler_combined = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")


# In[47]:


# ==================== 1. 加载模型（无标准化器） ====================
# TODO: 请替换为您的模型文件(.pkl)路径
model_ct = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# TODO: 请替换为您的模型文件(.pkl)路径
model_pet = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# TODO: 请替换为您的模型文件(.pkl)路径
model_combined = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")

# TODO: 请替换为您的模型文件(.pkl)路径
model_total = joblib.load(r"REPLACE_WITH_YOUR_FILE.pkl")


# In[48]:


# ==================== 2. 读取测试集 ====================
# TODO: 请替换为您的CSV文件路径
ct_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
pet_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
combined_test = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")
# TODO: 请替换为您的CSV文件路径
clinic = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv")

# 合并临床数据
clinic_subset = clinic[['Patient_ID', 'MTV', 'label']]
total_test = combined_test.merge(clinic_subset, on='Patient_ID', how='inner')

# 提取标签（假设两个文件标签一致）
y_test = ct_test['label'].values

# 读取特征列表（与训练时相同）
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    ct_features = [line.strip() for line in f.readlines()]
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    pet_features = [line.strip() for line in f.readlines()]
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    combined_features = [line.strip() for line in f.readlines()]

# 确保特征存在于数据中
ct_features = [f for f in ct_features if f in ct_test.columns]
pet_features = [f for f in pet_features if f in pet_test.columns]
combined_features = [f for f in combined_features if f in combined_test.columns]
total_features = combined_features + ['MTV']
print(f"CT特征数：{len(ct_features)}")
print(f"PET特征数：{len(pet_features)}")
print(f"PET/CT特征数：{len(combined_features)}")
print(f"组学+临床特征数：{len(total_features)}")


# In[49]:


# # ==================== 3. 特征提取与标准化 ====================
# X_ct = ct_test[ct_features].values
# X_ct_scaled = scaler_ct.transform(X_ct)

# X_pet = pet_test[pet_features].values
# X_pet_scaled = scaler_pet.transform(X_pet)

# X_combined = combined_test[combined_features].values
# X_combined_scaled = scaler_combined.transform(X_combined)


# In[50]:


# ==================== 3. 特征提取（无标准化） ====================
X_ct = ct_test[ct_features].values
X_pet = pet_test[pet_features].values
X_combined = combined_test[combined_features].values
X_total = total_test[total_features].values
print(f"CT测试集样本数：{X_ct.shape[0]}")
print(f"PET测试集样本数：{X_pet.shape[0]}")
print(f"PET/CT测试集样本数：{X_combined.shape[0]}")
print(f"组学+临床测试集样本数：{X_total.shape[0]}")


# In[51]:


# # ==================== 4. 预测概率 ====================
# prob_ct = model_ct.predict_proba(X_ct_scaled)[:, 1]
# prob_pet = model_pet.predict_proba(X_pet_scaled)[:, 1]
# prob_combined = model_combined.predict_proba(X_combined_scaled)[:, 1]


# In[52]:


# ==================== 4. 预测概率（无标准化） ====================
prob_ct = model_ct.predict_proba(X_ct)[:, 1]
prob_pet = model_pet.predict_proba(X_pet)[:, 1]
prob_combined = model_combined.predict_proba(X_combined)[:, 1]
prob_total = model_total.predict_proba(X_total)[:, 1]


# In[53]:


# ==================== 5. 导出为 CSV ====================
df_export = pd.DataFrame({
    'true_label': y_test,
    'prob_CT': prob_ct,
    'prob_PET': prob_pet,
    'prob_Combined': prob_combined,
    'prob_total': prob_total
})


# In[54]:


# TODO: 请替换为您的CSV文件路径
df_export.to_csv(r"REPLACE_WITH_YOUR_FILE.csv", index=False, encoding='utf-8-sig')
print("已导出 proba_rf.csv，可直接导入 SPSS。")


# In[ ]:




