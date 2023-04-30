from hmm_taggin.newsrc import dataloader
from linear.src import lm

if __name__ == '__main__':
    training_path = '/Users/liguangyu/su_lzh/hmm_taggin/train.conll'
    dataset = dataloader.Loader(training_path)

    sent = dataset.sent_list[0]
    print(sent)
    lm = lm.linearModel(dataset)

    epsil = lm.create_feature_space()

    f = lm.ftpl_instantialize(sent, 0, 'NR')
    print(f)

    seq,f,fredict = lm.sequentialize(f)
    print(f)
    print(len(f))
    print(lm.d)
    a = []
    for i in f:
        if i > 0:
            a.append(i)
    print(a)