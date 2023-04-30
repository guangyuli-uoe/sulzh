from hmm_taggin.newsrc import dataloader
from linear.src import lm

if __name__ == '__main__':
    training_path = '/Users/liguangyu/su_lzh/hmm_taggin/train.conll'
    dataset = dataloader.Loader(training_path)

    sent = dataset.sent_list[0]
    print(sent)

    lm = lm.linearModel(dataset)


    # print(dataset.align_dict)

    epsil = lm.create_feature_space()
    # print(epsil)
    # print(len(epsil))
    # print(epsil.index('02_NN_说'))

    tmp = lm.ftpl_instantialize(sent, 1, 'NN')
    print(tmp)
    print(lm.sequentialize(tmp))

    sent_list = dataset.sent_list
    sent_tag_list = dataset.sent_tag_list

    a = set()
    b = []
    for sentid, sent in enumerate(sent_list):
        tag_list = sent_tag_list[sentid]
        print(sent)
        print(tag_list)
        for i,tag in enumerate(tag_list):
            tmp_features = lm.ftpl_instantialize(sent, i, tag)
            for f in tmp_features:
                a.add(f)
                b.append(f)

        break
    print(a)
    print(b)
    # print(b.index('02_NN_说'))

    sent1 = ['戴相龙', '说']
    tag1 = ['NR', 'VV']
    c = set()
    d = []
    for i, tag in enumerate(tag1):
        tmp_features = lm.ftpl_instantialize(sent1, i, tag)
        for f in tmp_features:
            c.add(f)
            d.append(f)
    # print(lm.ftpl_instantialize(sent1,))
    print(c)
    print(d)
