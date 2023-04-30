
import argparse
import random
from datetime import datetime, timedelta

import numpy as np

from config import Config
from corpus import Corpus

from lm_zy import LinearModel

if __name__ == '__main__':
    # config = Config(args.bigdata)

    training_path = '/Users/liguangyu/su_lzh/hmm_taggin/train.conll'
    dev_path = '/Users/liguangyu/su_lzh/hmm_taggin/dev.conll'

    corpus = Corpus(training_path)
    print(corpus)

    trainset = corpus.load(training_path)
    devset = corpus.load(dev_path)
    print("  size of trainset: %d\n"
          "  size of devset: %d" %
          (len(trainset), len(devset)))
    '''
    size of trainset: 803
    size of devset: 1910
    '''

    lm = LinearModel(corpus.nt)
    # print(lm.W)
    print(lm)

    lm.create_feature_space(trainset)
    print(lm.epsilon[0])
    # ('06', 19, (973, 70, 879), 42), ('14', 18, (1089,))
    '''
        # ('02', 19, (890, 1320))
        each instance
    '''
    print(len(lm.fdict)) # 81113
    count = 0
    for batch in trainset:
        count += 1
        print(batch)
        print(len(batch))
        print(len(batch[0]))
        print(len(batch[1]))

        if count >= 3:
            break

    # print(corpus.wdict) # '圣保罗': 1468
    # print(corpus.tdict)
    # print(corpus.cdict) # '钢': 1734

    # print(lm.fdict)

    lm.online(trainset=trainset,
              devset=devset,
              file='./',
              epochs=1,
              interval=10,
              average=False)