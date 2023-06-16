import torch
from src import dataloader
from src import crf05
# beta = torch.zeros(3, 4).fill_(-1e9)
# print(beta)

if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)

    my_crf = crf05.LinearChainCRF(train_dataset, dev_dataset)

    # sentence = train_dataset.sent_word_list[0]
    # print(sentence)

    # my_crf.viterbi_predict(sentence)

    my_crf.sgd_training()