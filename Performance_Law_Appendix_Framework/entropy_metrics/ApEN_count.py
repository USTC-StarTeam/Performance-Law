import numpy as np
import pandas as pd
from tqdm import tqdm

def ApEn(U, m, r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        C = np.array(C)
        C[C == 0] = 1e-10
        if N - m + 1.0 == 0.0:
            return 0.0
        else:
            return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

df = pd.read_csv('sasrec_format.csv')

m = 2  
r = 0.2 
for m in [1,2,4,8]:
    for n in [-1,0,0.1,1,2]:
        for seql in [25,50]:

            entropies = []
            for sequence in tqdm(df['sequence_item_ids'], desc="Calculating Approximate Entropy"):
                sequence = list(map(int, sequence.strip('"').split(',')))[-seql:]
                entropy = ApEn(sequence, m, n)
                entropies.append(entropy)

            mean_entropy = np.mean(entropies)

            print('m',m,'n',n,'seql',seql,"ApEN:", mean_entropy)
