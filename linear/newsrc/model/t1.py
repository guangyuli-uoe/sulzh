from hmm_taggin.newsrc import dataloader
# from linear.newsrc import lm
import lm425 as lm

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
    # print(len(epsil)) # 60721

    print(dataset.tag_list)

    lm.online_training(1)