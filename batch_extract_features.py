#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import logging
import SimpleITK as sitk
from radiomics import featureextractor


# In[10]:


# ---------- 配置日志 ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- 定义单病例提取函数（增加 config_file 参数） ----------
def extract_one_case(image_path: str, mask_path: str, config_file: str = None, label: int = 1):
    """
    为单个病例提取影像组学特征

    参数:
        image_path: 图像文件路径 (.nii.gz)
        mask_path:  掩膜文件路径 (.nii.gz)
        config_file: 特征提取配置文件路径 (YAML)。若为 None，则使用代码中默认路径。
        label:      掩膜中的标签值，默认为 1

    返回:
        dict: 包含所有特征（包括诊断信息）的字典
    """
    # 若未提供配置文件，则使用默认路径（请根据实际情况修改）
    if config_file is None:
        # config_file = r"REPLACE_WITH_YOUR_FILE.yaml"   # 修改为您的配置文件路径
        print('config_file is none')

    # 手动读取图像和掩膜（以便后续显式删除）
    img = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    extractor = featureextractor.RadiomicsFeatureExtractor(config_file)
    feature_dict = extractor.execute(img, mask, label=label)

    # 显式删除大对象并强制垃圾回收
    del img, mask, extractor
    import gc
    gc.collect()
    
    return feature_dict


# In[11]:


# ---------- 批量提取主函数 ----------
def batch_extract_radiomics(root_dir: str, output_csv: str, config_file: str = None, label: int = 1):
    """
    批量提取指定根目录下所有患者的影像组学特征

    参数:
# TODO: 请替换为您的实际目录路径
        root_dir:   根目录路径，例如 r"REPLACE_WITH_YOUR_DIRECTORY_PATH"
                    该目录下每个患者应为一个子文件夹，子文件夹内包含:
                        image/pet_resample.nii.gz
                        mask/pet_resample_mask.nii.gz
        output_csv: 输出CSV文件路径
        config_file: 特征提取配置文件路径（可选）
        label:      掩膜中的标签值，默认为1
    """
    # 获取所有患者子文件夹
    patient_dirs = [d for d in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, d))]
    logging.info(f"找到 {len(patient_dirs)} 个患者文件夹")

    results = []
    failed = []

    for patient_id in patient_dirs:
        # 构建图像和掩膜路径（请根据实际文件名调整）
        img_path = os.path.join(root_dir, patient_id, "image", "pet.nii.gz")
        mask_path = os.path.join(root_dir, patient_id, "mask", "pet_mask.nii.gz")

        # 检查文件是否存在
        if not os.path.isfile(img_path):
            logging.warning(f"跳过 {patient_id}: 图像文件不存在 -> {img_path}")
            failed.append(patient_id)
            continue
        if not os.path.isfile(mask_path):
            logging.warning(f"跳过 {patient_id}: 掩膜文件不存在 -> {mask_path}")
            failed.append(patient_id)
            continue

        try:
            logging.info(f"正在处理: {patient_id}")
            features = extract_one_case(img_path, mask_path, config_file, label)

            # 添加患者ID
            features['Patient_ID'] = patient_id

            results.append(features)
            logging.info(f"成功提取: {patient_id}")
        except Exception as e:
            logging.error(f"处理 {patient_id} 时发生异常: {e}")
            failed.append(patient_id)

    # 保存结果
    if not results:
        logging.warning("没有成功提取任何特征，程序退出。")
        return

    df = pd.DataFrame(results)
    # 将 'Patient_ID' 列移动到第一列
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Patient_ID')))
    df = df[cols]

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logging.info(f"特征提取完成！共成功 {len(results)} 例，失败 {len(failed)} 例。")
    logging.info(f"结果已保存至: {output_csv}")

    if failed:
        logging.info(f"失败患者列表: {failed}")


# In[12]:


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 请根据实际路径修改以下变量
# TODO: 请替换为您的实际目录路径
    root_directory = r"REPLACE_WITH_YOUR_DIRECTORY_PATH"          # 患者根目录
# TODO: 请替换为您的CSV文件路径
    output_file = r"REPLACE_WITH_YOUR_FILE.csv"  # 输出CSV路径
# TODO: 请替换为您的PyRadiomics参数配置文件(.yaml)路径
    config_yaml = r"REPLACE_WITH_YOUR_FILE.yaml"   # 配置文件路径

    batch_extract_radiomics(root_directory, output_file, config_file=config_yaml)

