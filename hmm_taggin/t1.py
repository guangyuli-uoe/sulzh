from collections import Counter
import numpy as np

a = {'a': 333, 'b': 2, 'c': 111}
print(sorted(a))
print(a['a'])


print(sorted(a.items(), key=lambda x: x[1], reverse=True))

dic={'a': 4, 'b': 3, 'c': 2, 'd': 1}
print({k:v for k,v in sorted(dic.items(), key=lambda x: x[1],reverse=True)})

tags = ['NN', 'PU', 'VV', 'AD','NN', 'PU', 'VV', 'AD']

rtag = 'NN'
ctag = 'PU'
print(tags)
print(tags[:-1])
print(tags[1:])

tmp_list = [pair for pair in zip(tags[:-1], tags[1:])]
print(Counter(tmp_list))



# def n_subseq(n):

# print(np.zeros([3, 3]))
npar = np.zeros([3, 3])
print(npar)
for i,x in enumerate(npar):
    for j,y in enumerate(x):
        npar[i,j] = i+j
print(npar)


aaa = [
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [1, 4, 4, 4],
    [1, 5, 5, 5],
    [1, 6, 6, 6]
]

print([i[0] for i in aaa])

tag1 = [['NR', 'VV', 'NR'], ['NR', 'VV', 'NR', 'NN'], ['NR', 'VV', 'NR'], ['NR', 'VV', 'NR']]
token1 = [['aa', 'love', 'you'], ['i', 'love', 'you', 'and'], ['i', 'love', 'you'], ['a', 'b', 'c']]

# aignment = [pair for pair in zip(tag1[j], token1[j]) for i in range(len(tag1))]
print([pair for pair in zip(tag1, token1)])
align = [pair for pair in zip(tag1, token1)]

count_align = 0
for tu in align:
    if 'NR' in tu[0]:
        if 'i' in tu[1]:
            if tu[1].index('i') == tu[0].index('NR'):
                count_align += 1
print(count_align)