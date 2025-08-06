import numpy as np
import pandas as pd
from tqdm import tqdm

# 定义近似熵函数
def ApEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        # 避免 C 中有零导致对数计算问题
        C = np.array(C)
        C[C == 0] = 1e-10
        if N - m + 1.0 == 0.0:
            return 0.0
        else:
            return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

# 读取CSV文件
df = pd.read_csv('sasrec_format.csv')

m = 2  # 嵌入维度，可以根据需要调整
r = 0.2  # 容差参数，一般取序列标准差的一定比例
for m in [1,2,4,8]:
    for n in [-1,0,0.1,1,2]:
        for seql in [25,50]:

            # 计算每条序列的近似熵
            entropies = []
            # 使用 tqdm 添加进度条
            for sequence in tqdm(df['sequence_item_ids'], desc="Calculating Approximate Entropy"):
                # 移除引号并分割成数字列表
                sequence = list(map(int, sequence.strip('"').split(',')))[-seql:]
                # 计算近似熵
                entropy = ApEn(sequence, m, n)
                entropies.append(entropy)

            # 计算近似熵的均值
            mean_entropy = np.mean(entropies)

            #print("每条序列的近似熵:", entropies)
            print('m',m,'n',n,'seql',seql,"近似熵的均值:", mean_entropy)