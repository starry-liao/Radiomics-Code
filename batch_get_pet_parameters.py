#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel as nib
import pandas as pd


# In[2]:


# ---------- 复用并调整单患者提取函数（添加verbose控制） ----------
def extract_pet_parameters_single(pet_path, mask_path, patient_id="Patient_001", verbose=False):
    """
    为单个患者提取PET参数（与原代码一致，增加verbose开关）
    
    参数:
        pet_path: PET图像文件路径 (.nii 或 .nii.gz)
        mask_path: 掩膜文件路径 (.nii 或 .nii.gz)
        patient_id: 患者ID
        verbose: 是否打印详细信息
    
    返回:
        dict: 包含所有PET参数的字典，失败返回None
    """
    try:
        if verbose:
            print("正在加载图像...")
        pet_img = nib.load(pet_path)
        mask_img = nib.load(mask_path)
        
        pet_data = pet_img.get_fdata()
        mask_data = mask_img.get_fdata()
        
        if verbose:
            print(f"PET图像尺寸: {pet_data.shape}")
            print(f"掩膜尺寸: {mask_data.shape}")
        
        mask_binary = (mask_data > 0).astype(np.uint8)
        roi_pet_values = pet_data[mask_binary > 0]
        
        if len(roi_pet_values) == 0:
            print(f"警告: 患者 {patient_id} 的掩膜内无体素，跳过")
            return None
        
        if verbose:
            print(f"ROI内体素数量: {len(roi_pet_values)}")
        
        affine = pet_img.affine
        voxel_volume_mm3 = np.abs(np.linalg.det(affine[:3, :3]))
        voxel_volume_cm3 = voxel_volume_mm3 / 1000
        
        if verbose:
            print(f"单个体素体积: {voxel_volume_mm3:.4f} mm³ = {voxel_volume_cm3:.6f} cm³")
        
        suv_max = np.max(roi_pet_values)
        suv_min = np.min(roi_pet_values)
        suv_mean = np.mean(roi_pet_values)
        suv_std = np.std(roi_pet_values)
        suv_median = np.median(roi_pet_values)
        
        mtv_cm3 = np.sum(mask_binary) * voxel_volume_cm3
        tlg = suv_mean * mtv_cm3
        
        # 简化版SUVpeak（与原代码一致）
        suv_peak = calculate_suv_peak_simple(pet_data, mask_binary)
        
        suv_percentiles = np.percentile(roi_pet_values, [10, 25, 50, 75, 90])
        
        pet_params = {
            'Patient_ID': patient_id,
            'SUVmax': round(suv_max, 4),
            'SUVmin': round(suv_min, 4),
            'SUVmean': round(suv_mean, 4),
            'SUVstd': round(suv_std, 4),
            'SUVmedian': round(suv_median, 4),
            'SUVpeak': round(suv_peak, 4),
            'SUV_10percentile': round(suv_percentiles[0], 4),
            'SUV_25percentile': round(suv_percentiles[1], 4),
            'SUV_75percentile': round(suv_percentiles[3], 4),
            'SUV_90percentile': round(suv_percentiles[4], 4),
            'MTV_cm3': round(mtv_cm3, 4),
            'TLG': round(tlg, 4),
            'Voxel_count': int(np.sum(mask_binary)),
            'Voxel_volume_mm3': round(voxel_volume_mm3, 6),
            'Voxel_volume_cm3': round(voxel_volume_cm3, 8)
        }
        
        return pet_params
        
    except Exception as e:
        print(f"处理患者 {patient_id} 时出错: {e}")
        return None


# In[3]:


def calculate_suv_peak_simple(pet_data, mask_data):
    """简化版SUVpeak计算 - 使用3x3x3体素区域的最大平均值"""
    try:
        roi_coords = np.argwhere(mask_data > 0)
        roi_values = pet_data[mask_data > 0]
        max_idx = np.argmax(roi_values)
        max_coord = roi_coords[max_idx]
        
        half_size = 1  # 3x3x3 的半宽
        z_start = max(0, max_coord[0] - half_size)
        z_end = min(pet_data.shape[0], max_coord[0] + half_size + 1)
        y_start = max(0, max_coord[1] - half_size)
        y_end = min(pet_data.shape[1], max_coord[1] + half_size + 1)
        x_start = max(0, max_coord[2] - half_size)
        x_end = min(pet_data.shape[2], max_coord[2] + half_size + 1)
        
        cube_region = pet_data[z_start:z_end, y_start:y_end, x_start:x_end]
        suv_peak = np.mean(cube_region)
        return suv_peak
    except Exception as e:
        print(f"计算SUVpeak时出错: {e}")
        return np.mean(pet_data[mask_data > 0])


# In[4]:


# ---------- 批量提取主函数 ----------
def batch_extract_pet_parameters(root_dir, output_csv):
    """
    批量提取指定根目录下所有患者的PET参数
    
    参数:
# TODO: 请替换为您的实际目录路径
        root_dir: 根目录路径，例如 r"REPLACE_WITH_YOUR_DIRECTORY_PATH"
        该目录下每个患者应为一个子文件夹，子文件夹内包含:
            image/pet.nii.gz
            mask/pet_mask.nii.gz
        output_csv: 输出CSV文件路径
    """
    results = []
    
    # 获取根目录下所有子文件夹（患者ID）
    patient_dirs = [d for d in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir, d))]
    
    print(f"找到 {len(patient_dirs)} 个患者文件夹，开始提取...")
    
    for patient_id in patient_dirs:
        pet_path = os.path.join(root_dir, patient_id, "image", "pet.nii.gz")
        mask_path = os.path.join(root_dir, patient_id, "mask", "pet_mask.nii.gz")
        
        # 检查文件是否存在
        if not os.path.isfile(pet_path):
            print(f"跳过 {patient_id}: PET文件不存在 ({pet_path})")
            continue
        if not os.path.isfile(mask_path):
            print(f"跳过 {patient_id}: 掩膜文件不存在 ({mask_path})")
            continue
        
        # 提取参数（verbose=False 减少输出）
        params = extract_pet_parameters_single(pet_path, mask_path, 
                                               patient_id=patient_id, 
                                               verbose=False)
        if params:
            results.append(params)
            print(f"成功提取: {patient_id}")
        else:
            print(f"提取失败: {patient_id}")
    
    if not results:
        print("没有成功提取到任何参数，请检查数据路径。")
        return
    
    # 保存为CSV
    df = pd.DataFrame(results)
    # 确保第一列为患者ID（已在DataFrame中）
    df.to_csv(output_csv, index=False)
    print(f"\n批量处理完成！共成功提取 {len(results)} 例，结果已保存至: {output_csv}")


# In[5]:


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 请根据实际路径修改
# TODO: 请替换为您的实际目录路径
    root_directory = r"REPLACE_WITH_YOUR_DIRECTORY_PATH"   # 良性患者根目录
# TODO: 请替换为您的CSV文件路径
    output_file = r"REPLACE_WITH_YOUR_FILE.csv"
    
    batch_extract_pet_parameters(root_directory, output_file)


# In[6]:


# TODO: 请替换为您的实际目录路径
root_directory = r"REPLACE_WITH_YOUR_DIRECTORY_PATH"   # 良性患者根目录
# TODO: 请替换为您的CSV文件路径
output_file = r"REPLACE_WITH_YOUR_FILE.csv"

batch_extract_pet_parameters(root_directory, output_file)


# In[ ]:




