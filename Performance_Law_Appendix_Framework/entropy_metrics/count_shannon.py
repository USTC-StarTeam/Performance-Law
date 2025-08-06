import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 定义香农熵函数
def shannon_entropy(sequence):
    # 计算值的频率
    value, counts = np.unique(sequence, return_counts=True)
    # 将频率转换为概率分布
    probabilities = counts / counts.sum()
    # 计算香农熵
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 设置工作参数
seql = 10000  # 每片段长度

# 设置文件路径
folder_path = '.'  # 假设在当前目录中
files = [f for f in os.listdir(folder_path) if f.startswith('sasrec_format_') and f.endswith('.csv')]

with open('output_shannon.txt', 'w') as output_file:
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # 判断数据集类型并设置不同的截取长度
        if 'krnd' in file:
            truncate_lengths = [25, 50, 100]
        elif 'ml1m' in file:
            truncate_lengths = [50, 100, 150, 200]
        else:
            truncate_lengths = [len(list(map(int, df['sequence_item_ids'].iloc[0].strip('"').split(','))))]
        
        for truncate_length in truncate_lengths:
            all_sequences = []
            
            # 读取和连接缩短的序列
            for sequence in df['sequence_item_ids']:
                sequence = list(map(int, sequence.strip('"').split(',')))
                all_sequences.extend(sequence[-truncate_length:])
            
            entropies = []
            # 计算并记录每个片段的香农熵
            for i in tqdm(range(0, len(all_sequences) - seql + 1, seql), desc=f"Calculating Shannon Entropy for {file} with truncate length {truncate_length}"):
                chunk = all_sequences[i:i + seql]
                entropy = shannon_entropy(chunk)
                entropies.append(entropy)
            
            mean_entropy = np.mean(entropies) if entropies else 0
            output_file.write(f"数据集 {file}, 裁剪长度 {truncate_length}, 每个片段的香农熵均值: {mean_entropy}\n")
            print(f"数据集 {file}, 裁剪长度 {truncate_length}, 每个片段的香农熵均值: {mean_entropy}")