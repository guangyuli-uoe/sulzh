from src import dataloader
from src import crf01

if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    # print(train_dataset.tag_dict)

    my_crf = crf01.LinearChainCRF(train_dataset, dev_dataset)
    # print(my_crf.epsilon)
    # print(my_crf.tag_dict)
    my_crf.sgd_training()