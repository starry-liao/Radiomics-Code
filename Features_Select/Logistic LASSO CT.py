#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ==========================================
# 影像组学特征筛选全流程（基于高 ICC 特征 + LASSO CV）
# 仅保留预处理 + LASSO 交叉验证
# 可视化：交叉验证误差曲线 + 系数路径图
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold, train_test_split


# In[2]:


# 设置绘图风格（期刊标准）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")


# In[3]:


# ==================== 1. 读取数据 ====================
# 读取训练集 CT 数据
# TODO: 请替换为您的CSV文件路径
df = pd.read_csv(r"REPLACE_WITH_YOUR_FILE.csv", encoding='utf-8-sig')
# 读取高 ICC 特征列表（每行一个特征名）
# TODO: 请替换为您的ICC特征文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r') as f:
    high_icc_features = [line.strip() for line in f.readlines()]

# 确保 Patient_ID 和 label 列存在
if 'Patient_ID' not in df.columns:
    raise ValueError("CSV 文件中缺少 Patient_ID 列")
if 'label' not in df.columns:
    raise ValueError("CSV 文件中缺少 label 列")

# 筛选出高 ICC 特征（仅保留存在于数据中的特征）
existing_features = [f for f in high_icc_features if f in df.columns]
missing = set(high_icc_features) - set(existing_features)
if missing:
    print(f"警告：以下高 ICC 特征在数据中不存在，已忽略：{missing}")

# 构建特征矩阵 X 和标签 y
X = df[existing_features].copy()
y = df['label'].copy()

print(f"原始样本数：{X.shape[0]}, 高 ICC 特征数：{X.shape[1]}")


# In[4]:


# ==================== 2. 预处理 ====================
# 2.1 缺失值处理：删除包含缺失值的样本（或可填充中位数）
if X.isnull().any().any():
    print("存在缺失值，将删除包含缺失值的样本")
    X = X.dropna()
    y = y.loc[X.index]
    print(f"删除后样本数：{X.shape[0]}")

# # 2.2 低方差过滤（阈值 1e-4）
# vt = VarianceThreshold(threshold=1e-4)
# X_vt = pd.DataFrame(vt.fit_transform(X), index=X.index,
#                     columns=X.columns[vt.get_support()])
# print(f"去低方差后特征数：{X_vt.shape[1]}")
X_vt = X
print(f"特征数：{X_vt.shape[1]}")


# In[5]:


# 2.3 高相关过滤（|r|>0.9 保留第一个）
corr = X_vt.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
X_corr = X_vt.drop(columns=to_drop)
print(f"去高相关后特征数：{X_corr.shape[1]}")


# In[6]:


# ==================== 单变量 Mann-Whitney U 初筛 ====================
def mann_p(x, y):
    g0 = x[y == 0]
    g1 = x[y == 1]
    if len(g0) == 0 or len(g1) == 0:
        return 1.0
    return mannwhitneyu(g0, g1, alternative='two-sided')[1]

pvals = X_corr.apply(lambda col: mann_p(col, y))
reject, p_corr, _, _ = multipletests(pvals, method='fdr_bh', alpha=0.05)
X_uni = X_corr.loc[:, reject]
print(f"Mann-Whitney U 初筛后特征数：{X_uni.shape[1]}")


# In[7]:


# ==================== 3. LASSO 交叉验证（L1 逻辑回归） ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_uni)

# 定义候选正则化参数 C（对数均匀分布）
Cs = np.logspace(-3, 1, 50)

lasso_cv = LogisticRegressionCV(
    cv=5, 
    penalty='l1', 
    solver='liblinear',
    Cs=Cs,
    scoring='roc_auc', 
    n_jobs=-1,
    random_state=42
)
lasso_cv.fit(X_scaled, y)

# 最佳 C 值
best_C = lasso_cv.C_[0]
print(f"最佳正则化参数 C = {best_C:.6f}")

# 提取非零系数对应的特征及系数
coef = lasso_cv.coef_[0]
selected_mask = coef != 0
selected_features = X_uni.columns[selected_mask].tolist()
selected_coefs = coef[selected_mask]

print(f"LASSO 筛选后特征数：{len(selected_features)}")
print("入选特征及其系数：")
for feat, c in zip(selected_features, selected_coefs):
    print(f"  {feat} {c:.3f}")


# In[8]:


#  快速验证性能
# ------------------------------------------------
X_final_df = X_uni[selected_features]      # DataFrame，列名完整

scaler_2 = StandardScaler()
X_scaled_2 = scaler_2.fit_transform(X_final_df)   # 列数与后续完全一致

# 3. 后续直接用这个 X_scaled 即可，无需再 transform
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled_2, y,
                                          test_size=0.3,
                                          random_state=42,
                                          stratify=y)

clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
clf.fit(X_tr, y_tr)
y_pred = clf.predict_proba(X_te)[:, 1]

print(f'Hold-out AUC: {roc_auc_score(y_te, y_pred):.3f}')


# In[9]:


# # 保存最终特征列表
# with open('optimal_features_lasso_ct.txt', 'w') as f:
#     for feat in selected_features:
#         f.write(feat + '\n')
# print("\n最优特征列表已保存至 optimal_features_lasso_ct.txt")


# In[10]:


# ==================== 4. 可视化（期刊质量标准） ====================
# 4.1 交叉验证误差曲线（修正版）

Cs_actual = lasso_cv.Cs_

# ✅ 自动识别正类
positive_class = sorted(lasso_cv.classes_)[-1]

scores = lasso_cv.scores_[positive_class]  # (n_folds, n_Cs)

# ✅ 正确维度
mean_auc = scores.mean(axis=0)
std_auc = scores.std(axis=0)

mean_error = 1 - mean_auc

plt.figure(figsize=(8, 6))
plt.semilogx(Cs_actual, mean_error, 'b-o', markersize=4, linewidth=1.5)
plt.fill_between(Cs_actual, mean_error - std_auc, mean_error + std_auc, alpha=0.2)

plt.axvline(x=best_C, linestyle='--')

plt.xlabel('Regularization parameter C')
plt.ylabel('1 - AUC (Error)')
plt.title('LASSO Cross-Validation Error Curve')

# ⚠️ 关键：是否反转轴
plt.gca().invert_xaxis()

plt.tight_layout()
plt.show()


# In[14]:


# 4.2 系数路径图（保留您原代码风格，横轴为 log10(C)）
C_grid = np.logspace(-3, 1, 50)              # 手动定义 C 值网格
coef_path = []
for c in C_grid:
    clf = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=42)
    clf.fit(X_scaled, y)
    coef_path.append(clf.coef_[0])
coef_path = np.array(coef_path)

plt.figure(figsize=(10, 6))

for i, feat in enumerate(X_uni.columns):
    # 只有最终系数非零的特征才显示图例
    label = feat if feat in selected_features else None
    plt.plot(np.log10(C_grid), coef_path[:, i], alpha=0.6, linewidth=1, label=label)


# plt.plot(np.log10(C_grid), coef_path, alpha=0.4, linewidth=0.8)   # 多条半透明曲线
plt.axvline(np.log10(best_C), color='r', linestyle='--', linewidth=1.5, label=f'Best C = {best_C:.6f}')
plt.xlabel('log10(C)', fontsize=12)
plt.ylabel('Coefficient value', fontsize=12)
plt.title('LASSO Coefficient Path', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
# plt.savefig(r"REPLACE_WITH_YOUR_FILE.pdf", dpi=300, bbox_inches='tight')
# plt.savefig(r"REPLACE_WITH_YOUR_FILE.png", dpi=300, bbox_inches='tight')
plt.show()


# In[12]:


# ==================== 可选步骤（注释） ====================
# 以下为可选特征筛选方法，按需取消注释
# 
# # 单变量 Mann-Whitney U 初筛
# from scipy.stats import mannwhitneyu
# from statsmodels.stats.multitest import multipletests
# def mann_p(x, y):
#     g0, g1 = x[y==0], x[y==1]
#     return mannwhitneyu(g0, g1, alternative='two-sided')[1]
# pvals = X_corr.apply(lambda col: mann_p(col, y))
# reject, p_corr, _, _ = multipletests(pvals, method='fdr_bh')
# X_uni = X_corr.loc[:, reject]
# print(f'Mann-Whitney 初筛后特征数：{X_uni.shape[1]}')
#
# # 稳定性选择（需要安装 stability-selection 包）
# from stability_selection import StabilitySelection
# stab_sel = StabilitySelection(base_estimator=LogisticRegression(penalty='l1', solver='liblinear'),
#                               lambda_name='C', lambda_grid=np.logspace(-3, 0, 30),
#                               n_bootstrap=100, sample_fraction=0.75, random_state=0, n_jobs=-1)
# stab_sel.fit(X_scaled, y)
# stab_score = pd.Series(stab_sel.stability_scores_.max(axis=1), index=X_corr.columns)
# selected_stab = stab_score[stab_score > 0.5].index
# print(f'稳定性选择后特征数：{len(selected_stab)}')
#
# # RFE 二次精选
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# rfe_est = LogisticRegression(penalty='l1', solver='liblinear', C=best_C)
# rfe = RFE(rfe_est, n_features_to_select=5, step=5)
# rfe.fit(X_scaled, y)
# rfe_sel = X_corr.columns[rfe.support_]
# print(f'RFE 精选后特征数：{len(rfe_sel)}')

