from dataloader import DataLoader
from lm import LinearModel

if __name__ == '__main__':
    training_path = './train.conll'
    training_dataset = DataLoader(training_path)
    print(training_dataset)
    dev_path = './dev.conll'
    dev_dataset = DataLoader(dev_path)
    print(dev_dataset)

    # print(len(training_dataset.char_dict))
    # print(len(training_dataset.char_fre_dict))
    #
    # print(len(training_dataset.word_dict))
    # print(len(training_dataset.word_fre_dict))
    #
    # print(len(training_dataset.tag_dict))
    # print(len(training_dataset.tag_fre_dict))
    #
    # print(training_dataset.char_dict)

    lm = LinearModel(training_dataset, dev_dataset)
    # print(len(lm.create_feature_space()))
    lm.create_feature_space()
    lm.online_training(epochs=20)

    # print(lm.model)

    '''
        evaluate
    '''


    # print(training_dataset.sent_tag_list[0])
