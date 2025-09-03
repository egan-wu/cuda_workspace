import math
import time

def naive_safe_softmax(data, scale_en=False):
    if scale_en:
        scale = math.sqrt(len(data))
        for i in range(len(data)):
            data[i] = data[i] / scale

    max_val = float('-inf')
    for x in data:
        max_val = max(max_val, x)

    sum_val = 0
    for x in data:
        sum_val += math.exp(x - max_val)

    softmax_val = []
    for x in data:
        s = math.exp(x - max_val) / sum_val
        softmax_val.append(s)

    return softmax_val


def online_safe_softmax(data):
    sum_val = 0
    old_max = float("-inf")
    for x in data:
        new_max = max(old_max, x)
        sum_val = sum_val * math.exp(old_max - new_max) + math.exp(x - new_max)
        old_max = new_max

    softmax_val = []
    for x in data:
        s = math.exp(x - old_max) / sum_val
        softmax_val.append(s)

    return softmax_val

data_size = 10000000
data = [x for x in range(data_size)]

start = time.time()
naive_softmax_val = naive_safe_softmax(data)
# print(naive_softmax_val)
end = time.time()
print(f"naive_safe_softmax time: {end - start:.6f} sec")

start = time.time()
online_softmax_val = online_safe_softmax(data)
# print(online_softmax_val)
end = time.time()
print(f"online_safe_softmax time: {end - start:.6f} sec")

for i in range(data_size):
    if not math.isclose(naive_softmax_val[i], online_softmax_val[i], rel_tol=1e-7):
        print("isclose: fail!")
        break
else:
    print("isclose: pass!")