from src import dataloader
from src import crf04

if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)

    my_crf = crf04.LinearChainCRF(train_dataset, dev_dataset)
    # print(my_crf.bigram_features)
    my_crf.sgd_training()