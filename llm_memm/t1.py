from src import dataloader
from src import llm

def sequentialze(org_list, vocab_dict):
    new_list = []
    for element in org_list:
        element_id = vocab_dict[element]

        new_list.append(element_id)
    return new_list

def seq2(org_list, vocab_dict):

    return [vocab_dict[i] for i in org_list]

if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)

    sent_word_list = train_dataset.sent_word_list
    sent_tag_list = train_dataset.sent_tag_list
    sent_char_list = train_dataset.sent_char_list
    print(train_dataset.tag_dict)

    # print(sent_word_list)
    # print(train_dataset.word_dict)

    print(sent_word_list[0])
    sequentialzed_list = sequentialze(sent_word_list[0], train_dataset.word_dict)
    print(sequentialzed_list)
    print(len(sent_word_list[0]))
    print(len(sequentialzed_list))

    print(seq2(sent_word_list[0], train_dataset.word_dict))

    print(train_dataset.seqlized_sent_word_list[0])

    '''
        '08_AD_遍'
        ('15', 27, (370,))
    '''


    myllm = llm.LogLinearModel(train_dataset, dev_dataset)
    # myllm.create_feature_space()
    # print(myllm.epsilon)
    # print(len(myllm.epsilon))
    myllm.online_training(epochs=1)