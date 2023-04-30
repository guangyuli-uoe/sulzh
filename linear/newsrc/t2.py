import numpy as np


a = [1, 2, 3]
b = [1, 1, 1]

print(np.dot(a, b))

score = 0
for i in range(3):
    if i > score:
        score = i
print(score)

