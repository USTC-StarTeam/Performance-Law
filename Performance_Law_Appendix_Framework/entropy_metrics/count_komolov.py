import numpy as np
import pandas as pd
import os
import zlib
from tqdm import tqdm

# Define compression ratio function
def compression_ratio(data):
    # Convert the list of integers to a binary string
    byte_data = b''.join(int(x).to_bytes(4, 'big') for x in data)
    # Compress the binary data using zlib
    compressed_data = zlib.compress(byte_data)
    # Calculate and return the compression ratio
    return len(compressed_data) / len(byte_data)

# Configuration parameters
seql = 10000  # Length of each segment to compress

# File path settings
folder_path = '.'  
files = [f for f in os.listdir(folder_path) if f.startswith('sasrec_format_') and f.endswith('.csv')]

with open('output_k.txt', 'w') as output_file:
    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # Determine the truncation lengths based on dataset type
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
            
            compression_ratios = []
            # Calculate and record the compression ratio for each segment
            for i in tqdm(range(0, len(all_sequences) - seql + 1, seql), desc=f"Calculating Compression Ratio for {file} with truncate length {truncate_length}"):
                chunk = all_sequences[i:i + seql]
                ratio = compression_ratio(chunk)
                compression_ratios.append(ratio)
            
            mean_compression_ratio = np.mean(compression_ratios) if compression_ratios else float('inf')
            output_file.write(f"Dataset {file}, Seq_len {truncate_length},compression ratio: {mean_compression_ratio}\n")
            print(f"dataset {file}, Seq_len {truncate_length}, compression ratio: {mean_compression_ratio}")
