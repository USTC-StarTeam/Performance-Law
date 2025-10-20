import math
import numpy as np
import zlib

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = [
    'actual_entropy',
    'actual_entropy_tq',
    'ApEn',
    'compression_ratio',
    'shannon_entropy'
]

def actual_entropy(seq):
    n = len(seq)
    discovered = set()
    sum_lambda = 0
    for i in range(n):
        for j in range(i + 1, n+1):
            s = tuple(seq[i:j])
            if s not in discovered:
                sum_lambda += len(s)
                discovered.add(s)
                break
            elif j == n:
                sum_lambda += n - i + 1
                discovered.add(s)
    if n == 0: return 0.0
    return (1 / (sum_lambda / n)) * math.log(n)

def actual_entropy_tq(seq):
    if tqdm is None:
        raise ImportError("Please install tqdm: pip install tqdm")
    n = len(seq)
    discovered = set()
    sum_lambda = 0
    for i in tqdm(range(n), desc="Calculating Actual Entropy"):
        for j in range(i + 1, n+1):
            s = tuple(seq[i:j])
            if s not in discovered:
                sum_lambda += len(s)
                discovered.add(s)
                break
            elif j == n:
                sum_lambda += n - i + 1
                discovered.add(s)
    if n == 0: return 0.0
    return (1 / (sum_lambda / n)) * math.log(n)

def ApEn(U, m, r):
    """
    Approximate Entropy (ApEn) of a sequence U.
    m: Embedding dimension.
    r: Tolerance (similarity threshold).
    Returns a float value.
    """
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[U[j] for j in range(i, i + m)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        C = np.array(C)
        C[C == 0] = 1e-10  # avoid log(0)
        if N - m + 1.0 == 0.0:
            return 0.0
        else:
            return (N - m + 1.0)**(-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m+1) - _phi(m))

def compression_ratio(data):
    """
    Computes compression ratio for a list of integers using zlib.
    Ratio < 1 indicates compressible data, Ratio ~ 1 means nearly random data.
    """
    byte_data = b''.join(int(x).to_bytes(4, 'big') for x in data)
    compressed_data = zlib.compress(byte_data)
    return len(compressed_data) / len(byte_data)

def shannon_entropy(sequence):
    """
    Computes the Shannon entropy of a sequence.
    Returns entropy value in bits.
    """
    value, counts = np.unique(sequence, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy