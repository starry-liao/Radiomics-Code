#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from pingouin import intraclass_corr  # 需要安装 pingouin: pip install pingouin
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# ==================== 1. 读取数据 ====================
# TODO: 请替换为您的CSV文件路径
first_df = pd.read_csv('REPLACE_WITH_YOUR_FILE.csv')
# TODO: 请替换为您的CSV文件路径
second_df = pd.read_csv('REPLACE_WITH_YOUR_FILE.csv')

# 确保患者 ID 列名称一致（示例中为 Patient_ID）
pid_col = 'Patient_ID'
label_col = 'label'

# 按 Patient_ID 排序并合并，确保两次测量对应同一患者
first_df = first_df.sort_values(by=pid_col).reset_index(drop=True)
second_df = second_df.sort_values(by=pid_col).reset_index(drop=True)

# 检查患者 ID 是否完全一致
if not first_df[pid_col].equals(second_df[pid_col]):
    raise ValueError("两个文件的 Patient_ID 不一致，请检查数据！")

# 提取标签（仅用于参考，不参与 ICC 计算）
label = first_df[label_col]  # 假设两次分割标签相同

# ==================== 2. 筛选有效特征列 ====================
# 排除 ID 列和标签列，剩下的列为候选特征
feature_candidates = first_df.columns.drop([pid_col, label_col])

def is_numeric_series(series):
    """判断一个 Series 是否全部可转换为数值（忽略缺失值）"""
    try:
        pd.to_numeric(series, errors='raise')
        return True
    except (ValueError, TypeError):
        return False

valid_features = []
for col in feature_candidates:
    # 检查 first 和 second 中该列是否均为数值
    if is_numeric_series(first_df[col]) and is_numeric_series(second_df[col]):
        # 可选：进一步检查是否全为有限数值（无 inf 或 NaN）
        if np.isfinite(pd.to_numeric(first_df[col])).all() and \
           np.isfinite(pd.to_numeric(second_df[col])).all():
            valid_features.append(col)
        else:
            print(f"特征 {col} 包含无穷或缺失值，已剔除")
    else:
        print(f"特征 {col} 包含非数值内容（如版本、矩阵等），已剔除")

print(f"原始特征数: {len(feature_candidates)}, 有效数值特征数: {len(valid_features)}")


# In[ ]:


# ==================== 3. 计算每个特征的 ICC ====================
# 使用 ICC(2,1) 模型：绝对一致性，随机效应（适用于两次测量的一致性评估）
# 也可选用 ICC(3,1) 固定效应，此处使用 ICC(2,1) 为常用选择
def compute_icc_for_feature(feature_name):
    """计算单个特征在两次分割间的 ICC(2,1)"""
    # 构造长格式数据：每个患者有两行测量
    data = []
    for patient_id in first_df[pid_col]:
        # 第一次测量
        val1 = pd.to_numeric(first_df.loc[first_df[pid_col]==patient_id, feature_name].values[0])
        # 第二次测量
        val2 = pd.to_numeric(second_df.loc[second_df[pid_col]==patient_id, feature_name].values[0])
        data.append([patient_id, 1, val1])  # 测量次数 1 代表第一次分割
        data.append([patient_id, 2, val2])  # 测量次数 2 代表第二次分割
    
    long_df = pd.DataFrame(data, columns=['Patient_ID', 'rater', 'value'])
    # 计算 ICC(2,1) 模型：targets = Patient_ID, raters = rater, ratings = value
    icc_result = intraclass_corr(data=long_df, targets='Patient_ID', raters='rater', ratings='value')
    # icc_result 包含多行，其中 type 为 'ICC(2,1)' 对应的 ICC 值
    icc_val = icc_result.loc[icc_result['Type'] == 'ICC2', 'ICC'].values[0]
    return icc_val

# 存储所有特征的 ICC 结果
icc_results = []
for feat in valid_features:
    try:
        icc = compute_icc_for_feature(feat)
        icc_results.append((feat, icc))
    except Exception as e:
        print(f"计算特征 {feat} 时出错: {e}")
        icc_results.append((feat, np.nan))

# 转换为 DataFrame 并排序
icc_df = pd.DataFrame(icc_results, columns=['Feature', 'ICC'])
icc_df = icc_df.sort_values('ICC', ascending=False).reset_index(drop=True)

# ==================== 4. 筛选高一致性特征 ====================
threshold = 0.75  # 常用阈值，可根据需要调整
high_icc_features = icc_df[icc_df['ICC'] >= threshold]['Feature'].tolist()

print("\n=== 所有特征的 ICC 值（前20个）===")
print(icc_df.head(20))
print(f"\n共有 {len(high_icc_features)} 个特征的 ICC >= {threshold}")
print("高一致性特征列表（前10个）:", high_icc_features[:10])


# In[ ]:


# 保存结果到文件
# TODO: 请替换为您的CSV文件路径
icc_df.to_csv('REPLACE_WITH_YOUR_FILE.csv', index=False)
# TODO: 请替换为您的文本/特征文件路径
with open('REPLACE_WITH_YOUR_FILE.txt', 'w') as f:
    for feat in high_icc_features:
        f.write(feat + '\n')

print("\n结果已保存至 icc_results.csv 和 high_icc_features.txt")

