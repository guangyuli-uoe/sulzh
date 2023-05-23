from newsrc import dataloader
from newsrc import glm


if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    print(train_dataset.tag_dict)
    # print(train_dataset.sent_word_list[0])
    # print(train_dataset.sent_tag_list[0])


    myglm = glm.GlobalLinearModel(train_dataset, dev_dataset)
    # print(myglm.epsilon)
    # print(myglm.tag_dict)
    myglm.online_training(1)