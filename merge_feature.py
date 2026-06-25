#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 读取 PET 特征并添加前缀
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r', encoding='utf-8') as f:
    pet_features = ['PET-' + line.strip() for line in f if line.strip()]

# 读取 CT 特征并添加前缀
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'r', encoding='utf-8') as f:
    ct_features = ['CT-' + line.strip() for line in f if line.strip()]

# 合并所有特征
all_features = pet_features + ct_features

# 保存到新文件
# TODO: 请替换为您的特征列表文件(.txt)路径
with open(r"REPLACE_WITH_YOUR_FILE.txt", 'w', encoding='utf-8') as f:
    for feature in all_features:
        f.write(feature + '\n')

print(f"已保存 {len(all_features)} 个特征到 combined_features.txt")
print(f"其中 PET 特征 {len(pet_features)} 个，CT 特征 {len(ct_features)} 个")


# In[ ]:





# In[ ]:




