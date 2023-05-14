# global linear model
模型一次性考虑整个句子对应的词性序列(词性之间互相影响)，而不是刻画一个词 的词性，所以称为全局线性模型(Global Linear Model)
## comparison with linear model
+ linear model: 假设词语之间的词性相互独立预测
+ global linear model: 假设词语之间的词性是有关联的

# viterbi predict

# online training

# result
```commandline

DataLoader(
  num of sentences: 803
  num of uniq words: 4537
  num of uniq tags: 31
  num of uniq chars: 1896
)
DataLoader(
  num of sentences: 1910
  num of uniq words: 8577
  num of uniq tags: 30
  num of uniq chars: 2524
)
{'AD': 0, 'AS': 1, 'BA': 2, 'CC': 3, 'CD': 4, 'CS': 5, 'DEC': 6, 'DEG': 7, 'DER': 8, 'DEV': 9, 'DT': 10, 'ETC': 11, 'FW': 12, 'JJ': 13, 'LB': 14, 'LC': 15, 'M': 16, 'MSP': 17, 'NN': 18, 'NR': 19, 'NT': 20, 'OD': 21, 'P': 22, 'PN': 23, 'PU': 24, 'SB': 25, 'SP': 26, 'VA': 27, 'VC': 28, 'VE': 29, 'VV': 30}
---feature space constructing over---
---training set---
acc: 0.8914637723672632
---dev set---
acc: 0.7919871221606153
---training set---
acc: 0.9341449105309475
---dev set---
acc: 0.8310379777022596
---training set---
acc: 0.952723183729344
---dev set---
acc: 0.8475526143206343
---training set---
acc: 0.9615723085948958
---dev set---
acc: 0.8468570520081877
---training set---
acc: 0.9652390730419478
---dev set---
acc: 0.8474929946938532
---training set---
acc: 0.9749682213747922
---dev set---
acc: 0.8548460819968601
---training set---
acc: 0.968954727681627
---dev set---
acc: 0.8506727081221805
---training set---
acc: 0.9742837586780092
---dev set---
acc: 0.8533357181184046
---training set---
acc: 0.9729637234770705
---dev set---
acc: 0.8475128679027802
---training set---
acc: 0.9777549623545517
---dev set---
acc: 0.8418887497764264
---training set---
acc: 0.9815195071868583
---dev set---
acc: 0.8558596156521393
---training set---
acc: 0.9791238877481178
---dev set---
acc: 0.8529581271487907
---training set---
acc: 0.9780483035103158
---dev set---
acc: 0.8559192352789204
---training set---
acc: 0.9787816563997263
---dev set---
acc: 0.848287923050935
---training set---
acc: 0.9746259900264007
---dev set---
acc: 0.8369800671714461
---training set---
acc: 0.9773638408135328
---dev set---
acc: 0.8419483694032075
---training set---
acc: 0.9808839346827026
---dev set---
acc: 0.8447902382797751
---training set---
acc: 0.9802483621785469
---dev set---
acc: 0.8476122339474155
---training set---
acc: 0.9808350444900753
---dev set---
acc: 0.8455454202190028
---training set---
acc: 0.981470616994231
---dev set---
acc: 0.846558953874282
---online training over---
```