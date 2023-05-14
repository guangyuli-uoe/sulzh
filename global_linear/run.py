from src import dataloader
from src import glm
from src import glm02
import numpy as np



if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    print(train_dataset.tag_dict)

    # print({v: k for k, v in train_dataset.tag_dict.items()})

    myglm02 = glm02.GlobalLinearModel(train_dataset, dev_dataset)
    myglm02.online_training(20)