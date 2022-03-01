import gc
from numpy import delete


def next_power_two(n):  # 返回 >= n 的第一个2^k
    if n & (n - 1) == 0:
        return n
    return 1 << len(bin(n)[2:])


# def free_object(o):
#     delete o
#     gc.collect()
