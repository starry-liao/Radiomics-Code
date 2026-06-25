# Radiomics-Code

基于 PET/CT 影像组学的机器学习分析流程，用于医学图像特征提取、特征筛选、建模与评估。

所有硬编码的文件路径已替换为占位符并添加了中文注释说明。

## 目录结构

```
RadiomicsCode/
├── batch_extract_features.py      # 批量提取影像组学特征（PyRadiomics）
├── batch_get_patient_info.py      # 批量提取患者基本信息（DICOM）
├── batch_get_pet_parameters.py    # 批量提取PET定量参数（SUVmax, MTV, TLG等）
├── clinic.py                      # 临床指标单变量ROC分析
├── ICC.py                         # 组学特征一致性检验（ICC计算）
├── merge_feature.py               # 合并PET与CT特征列表
├── export proba for delong.py     # 导出模型预测概率（用于DeLong检验）
├── 决策曲线分析.py                  # 决策曲线分析（DCA）
├── 校准曲线.py                      # 校准曲线（Calibration Curve）
├── Features_Select/
│   ├── Logistic LASSO CT.py       # CT特征LASSO筛选
│   ├── Logistic LASSO PET.py      # PET特征LASSO筛选
│   └── Logistic LASSO Combined.py # CT+PET联合特征LASSO筛选
├── LR/
│   ├── LR_clinic.py               # 逻辑回归 - 临床模型
│   ├── LR_CT.py                   # 逻辑回归 - CT组学模型
│   ├── LR_PET.py                  # 逻辑回归 - PET组学模型
│   ├── LR_Combined.py             # 逻辑回归 - CT+PET联合模型
│   └── LR_组学+临床.py             # 逻辑回归 - 组学+临床融合模型
├── RF/
│   ├── RF_clinic.py               # 随机森林 - 临床模型
│   ├── RF_CT.py                   # 随机森林 - CT组学模型
│   ├── RF_PET.py                  # 随机森林 - PET组学模型
│   ├── RF_Combined.py             # 随机森林 - CT+PET联合模型
│   └── RF_组学+临床.py             # 随机森林 - 组学+临床融合模型
├── SVM/
│   ├── SVM_clinic.py              # 支持向量机 - 临床模型
│   ├── SVM_CT.py                  # 支持向量机 - CT组学模型
│   ├── SVM_PET.py                 # 支持向量机 - PET组学模型
│   ├── SVM_Combined.py            # 支持向量机 - CT+PET联合模型
│   └── SVM_组学+临床.py            # 支持向量机 - 组学+临床融合模型
└── ROC/
    ├── LR模型比较.py               # 逻辑回归多模型ROC比较
    ├── RF模型比较.py               # 随机森林多模型ROC比较
    ├── SVM模型比较.py              # SVM多模型ROC比较
    └── TOTAL模型比较.py            # 三种算法总体ROC比较
```

## 分析流程

```
DICOM数据 ──→ 患者信息提取 ──→ 临床指标分析
    │
    └──→ PET/CT图像 ──→ 组学特征提取（PyRadiomics）──→ ICC一致性检验
                                    │
                                    └──→ LASSO特征筛选 ──→ 建模（LR / RF / SVM）
                                                              │
                                                              └──→ ROC评估
                                                              └──→ 校准曲线
                                                              └──→ 决策曲线分析
```

## 依赖环境

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install SimpleITK pyradiomics nibabel pydicom
pip install pingouin statsmodels joblib
```

| 包 | 用途 |
|---|------|
| `pandas`, `numpy` | 数据处理 |
| `scikit-learn` | 机器学习（LR, RF, SVM, 标准化, 网格搜索） |
| `scipy` | 统计检验（Mann-Whitney U, DeLong检验） |
| `matplotlib`, `seaborn` | 数据可视化 |
| `SimpleITK`, `nibabel` | 医学图像读取（DICOM, NIfTI） |
| `pyradiomics` | 影像组学特征提取 |
| `pydicom` | DICOM元数据读取 |
| `pingouin` | ICC组内相关系数 |
| `statsmodels` | Lowess平滑, 多重检验校正 |
| `joblib` | 模型持久化保存/加载 |

## 使用方法

### 1. 克隆仓库

```bash
git clone https://github.com/starry-liao/Radiomics-Code.git
cd Radiomics-Code
```

### 2. 替换文件路径占位符

代码中的所有文件路径已替换为占位符，每个占位符上方均有 `# TODO:` 注释说明应替换为何种文件。

常见占位符：

| 占位符 | 应替换为 |
|--------|---------|
| `REPLACE_WITH_YOUR_FILE.csv` | CSV数据文件路径 |
| `REPLACE_WITH_YOUR_FILE.pkl` | 训练好的模型文件路径 |
| `REPLACE_WITH_YOUR_FILE.txt` | 特征列表文本文件路径 |
| `REPLACE_WITH_YOUR_FILE.yaml` | PyRadiomics参数配置文件路径 |
| `REPLACE_WITH_YOUR_DIRECTORY_PATH` | 数据文件夹路径 |

在IDE中使用全局搜索替换（Ctrl+Shift+H）即可快速定位并替换所有占位符。

### 3. 按顺序运行

建议按以下顺序运行脚本：

1. **数据提取**：`batch_get_patient_info.py` → `batch_get_pet_parameters.py` → `batch_extract_features.py`
2. **特征处理**：`ICC.py` → `merge_feature.py`
3. **特征筛选**：`Features_Select/` 目录下的 LASSO 脚本
4. **建模训练**：`LR/`、`RF/`、`SVM/` 目录下的模型脚本
5. **模型评估**：`ROC/` → `校准曲线.py` → `决策曲线分析.py`
6. **统计分析**：`clinic.py` → `export proba for delong.py`

### 4. 模型说明

- **临床模型**（`*_clinic.py`）：仅使用临床指标（如MTV）进行建模
- **CT模型**（`*_CT.py`）：仅使用CT影像组学特征
- **PET模型**（`*_PET.py`）：仅使用PET影像组学特征
- **联合模型**（`*_Combined.py`）：融合CT与PET组学特征
- **组学+临床模型**（`*_组学+临床.py`）：融合组学特征与临床指标

### 5. 模型评估指标

- **ROC曲线**与AUC（含95%置信区间，DeLong方法）
- **混淆矩阵**：灵敏度（Sensitivity）、特异度（Specificity）、准确率（Accuracy）
- **校准曲线**（Calibration Curve）与Brier Score
- **决策曲线分析**（Decision Curve Analysis, DCA）
- **分类报告**：Precision, Recall, F1-score

## 注意事项

- 本代码由Jupyter Notebook自动转换生成（`jupyter nbconvert --to script`），每段代码以 `# In[n]:` 标记为界，对应原notebook中的cell编号
- 所有硬编码路径已替换为占位符，运行前**务必**替换为实际文件路径
- 图像文件期望格式为 `.nii.gz`（NIfTI压缩格式）
- DICOM文件期望文件名为 `IM0`（无扩展名）
- 标签列名固定为 `label`，其中 `0` = 良性，`1` = 恶性
- 患者ID列名固定为 `Patient_ID`

## 引用

特征提取基于 [PyRadiomics](https://github.com/AIM-Harvard/pyradiomics)：
> van Griethuysen, J. J. M., et al. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. *Cancer Research*, 77(21), e104–e107.
