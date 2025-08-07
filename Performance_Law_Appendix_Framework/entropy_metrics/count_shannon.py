import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def shannon_entropy(sequence):
    value, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

seql = 10000  


folder_path = '.'  
files = [f for f in os.listdir(folder_path) if f.startswith('sasrec_format_') and f.endswith('.csv')]

with open('output_shannon.txt', 'w') as output_file:
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))

        if 'krnd' in file:
            truncate_lengths = [25, 50, 100]
        elif 'ml1m' in file:
            truncate_lengths = [50, 100, 150, 200]
        else:
            truncate_lengths = [len(list(map(int, df['sequence_item_ids'].iloc[0].strip('"').split(','))))]
        
        for truncate_length in truncate_lengths:
            all_sequences = []
            
            for sequence in df['sequence_item_ids']:
                sequence = list(map(int, sequence.strip('"').split(',')))
                all_sequences.extend(sequence[-truncate_length:])
            
            entropies = []
            for i in tqdm(range(0, len(all_sequences) - seql + 1, seql), desc=f"Calculating Shannon Entropy for {file} with truncate length {truncate_length}"):
                chunk = all_sequences[i:i + seql]
                entropy = shannon_entropy(chunk)
                entropies.append(entropy)
            
            mean_entropy = np.mean(entropies) if entropies else 0
            output_file.write(f"dataset {file}, truncate_length {truncate_length}, mean_entropy: {mean_entropy}\n")
            print(f"dataset {file}, truncate_length {truncate_length}, mean_entropy: {mean_entropy}")
