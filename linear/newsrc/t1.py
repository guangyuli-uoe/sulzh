from hmm_taggin.newsrc import dataloader
from linear.newsrc import lm

if __name__ == '__main__':
    training_path = '/Users/liguangyu/su_lzh/hmm_taggin/train.conll'
    dataset = dataloader.Loader(training_path)

    sent = dataset.sent_list[0]
    tag = dataset.sent_tag_list[0]
    print(sent)
    print(tag)
    lm = lm.linearModel(dataset)

    epsil = lm.create_feature_space()

    # print(epsil)

    f = lm.ftpl_instantialize(sent, 0, 'NR')
    print(f)

    seqlized_vec,final_vec,fre_dict = lm.sequentialize(f)
    print(seqlized_vec)
    print(len(seqlized_vec))
    # print(len(sent))
    print(final_vec)
    print(len(final_vec))

    print(dataset.tag_list)
    print(dataset.N)
    print(len(dataset.tag_list))

    print(dataset.id2tag(0))

    print(dataset.sent_list[0])



    # print(len(dataset.sent_list))

    # w = lm.online_training(1)
    # print(w)

