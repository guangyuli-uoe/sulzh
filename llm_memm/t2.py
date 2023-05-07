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

    train_dataset = dataloader.DataLoader(train_path)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path)

    sent_word_list = train_dataset.sent_word_list
    sent_tag_list = train_dataset.sent_tag_list
    sent_char_list = train_dataset.sent_char_list

    print(sent_word_list[0])
    print(sent_tag_list[0])

    myllm = llm.LogLinearModel(train_dataset, dev_dataset)
    # myllm.create_feature_space()

    fv = myllm.ftpl_instantialize(sent_word_list[0], 0, 'NR')
    print(fv)

    print(myllm.epsilon['02_NR_戴相龙'])
    # print(myllm.epsilon)

    print(myllm.W)

    print(myllm.scorer(fv))