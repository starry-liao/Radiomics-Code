#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import csv
import pydicom
from datetime import datetime


# In[6]:


def get_patient_info_from_dicom(dicom_file):
    """
    从单个DICOM文件提取患者基本信息
    """
    try:
        ds = pydicom.dcmread(dicom_file)
        
        # 提取基本信息
        name = str(getattr(ds, 'PatientName', '未知'))
        sex = getattr(ds, 'PatientSex', '未知')
        
        # 计算年龄
        birth_date = getattr(ds, 'PatientBirthDate', '')
        study_date = getattr(ds, 'StudyDate', '')
        age = "未知"
        if birth_date and study_date:
            try:
                birth = datetime.strptime(birth_date, '%Y%m%d')
                study = datetime.strptime(study_date, '%Y%m%d')
                age = study.year - birth.year
                if (study.month, study.day) < (birth.month, birth.day):
                    age -= 1
                age = f"{age}"
            except:
                pass
        
        # 身高体重
        height = getattr(ds, 'PatientSize', '未知')
        weight = getattr(ds, 'PatientWeight', '未知')
        
        return {
            'name': name,
            'sex': sex,
            'age': age,
            'height': height,
            'weight': weight
        }
        
    except Exception as e:
        print(f"读取DICOM文件失败 {dicom_file}: {e}")
        return None


# In[7]:


def batch_extract_patient_info(root_folder, output_csv):
    """
    批量提取malignant文件夹下所有患者的DICOM信息并保存到CSV
    - 文件存在则追加数据（不含表头）
    - 文件不存在则创建文件并写入表头
    """
    patient_records = []
    
    # 遍历benign/malignant文件夹下的所有子文件夹
    for patient_folder in os.listdir(root_folder):
        patient_path = os.path.join(root_folder, patient_folder)
        
        # 检查是否为文件夹
        if os.path.isdir(patient_path):
            patient_id = patient_folder  # 使用文件夹名作为患者编号
            
            # 查找DICOM文件 (假设文件名为IM0)
            dicom_file = os.path.join(patient_path, 'IM0')
            
            # 如果IM0不存在
            if not os.path.exists(dicom_file):
                print(f"未找到DICOM文件: {patient_path}")
                   
            
            # 提取患者信息
            info = get_patient_info_from_dicom(dicom_file)
            if info:
                record = {
                    'patient_id': patient_id,
                    'name': info['name'],
                    'sex': info['sex'],
                    'age': info['age'],
                    'height': info['height'],
                    'weight': info['weight']
                }
                patient_records.append(record)
                print(f"已提取: {patient_id} - {info['name']}")
    
    # 保存到CSV（追加模式）
    if patient_records:
        file_exists = os.path.exists(output_csv)
        
        with open(output_csv, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id', 'name', 'sex', 'age', 'height', 'weight'])
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入数据
            writer.writerows(patient_records)
        
        if file_exists:
            print(f"\n成功追加 {len(patient_records)} 条记录到 {output_csv}")
        else:
            print(f"\n成功创建文件并保存 {len(patient_records)} 条记录到 {output_csv}")
    else:
        print("未找到任何患者记录")


# In[12]:


if __name__ == "__main__":
    # 请修改为你的实际路径
# TODO: 请替换为您的实际目录路径
    root_folder = r"REPLACE_WITH_YOUR_DIRECTORY_PATH"  # benign/malignant文件夹路径
# TODO: 请替换为您的CSV文件路径
    output_csv = r"REPLACE_WITH_YOUR_FILE.csv"  # 输出CSV文件路径
    
    batch_extract_patient_info(root_folder, output_csv)


# In[14]:


# TODO: 请替换为您的实际目录路径
root_folder = r"REPLACE_WITH_YOUR_DIRECTORY_PATH"  # benign/malignant文件夹路径
# TODO: 请替换为您的CSV文件路径
output_csv = r"REPLACE_WITH_YOUR_FILE.csv"  # 输出CSV文件路径

batch_extract_patient_info(root_folder, output_csv)


# In[ ]:




