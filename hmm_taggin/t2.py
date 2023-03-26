import numpy as np
# a = []
#
# for i in range(3):
#     a.extend([i for i in range(3)])
# print(a)

a = [1, 2, 3]

for j in range(1, len(a)):
    print(a[j])


print(np.argmax(a))

aa = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 0],
    [7, 8, 9, 0],
    [10, 11, 12, 0]
])

for i in range(len(aa[0])):
    print(aa[:,i])
print('----------------------')
for i in range(aa.shape[1]):
    print(aa[:,i])
    print(np.argmax(aa[:,i]))

print('------------------------')
gold = ['NR', 'DT', 'M', 'NN', 'AD', 'AD', 'VA', 'PU', 'NT', 'CC', 'NR', 'NN', 'NN', 'NN', 'PU']
pred = ['NN', 'LC', 'M', 'NN', 'AD', 'AD', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN', 'NN']

count = 0
for i in range(len(gold)):
    # print(f'gold: {gold[i]}')
    # print(f'pred: {pred[i]}')
    if gold[i] == pred[i]:
        print(1)
        count += 1
    else:
        print(2)

print(f'count: {count}')
print(count / len(gold))