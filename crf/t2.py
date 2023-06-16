import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(a.shape)
print(a[0])

print(a[:,0].shape)

print(a[:,0])
b = np.array([[1, 2],
              [3, 4]])
print(a[:,0] + b)
print(a[:,0].reshape(1, -1) + b)
print(a[:,0].reshape(-1, 1) + b)

# for i in range(len(a[0])):
#     print(a[i, 0])
print('---6-9---')
cc = np.array([[1, 2, 3],
              [4, 5, 6]])
# for i in range(len(cc[:,-1])):
#     print(i)
#     print(cc[i:, 0])
for i in range(3):
    print(cc[0,i])
print(cc[:,0])

aaa = [1, 2, 3]
bbb = [4, 5, 6]

print(aaa+bbb)