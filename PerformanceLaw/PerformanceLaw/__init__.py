import math

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

__all__ = ['actual_entropy', 'actual_entropy_tq']

def actual_entropy(seq):
    """
    计算序列的实际熵（Actual Entropy）。
    参数:
        seq (list/str): 输入序列。
    返回:
        float: 实际熵。
    """
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
    """
    计算序列的实际熵（Actual Entropy），带进度条。
    参数:
        seq (list/str): 输入序列。
    返回:
        float: 实际熵。
    """
    if tqdm is None:
        raise ImportError("请先安装 tqdm: pip install tqdm")
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