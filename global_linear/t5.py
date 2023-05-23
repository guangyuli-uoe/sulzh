from fsrc import dataloader
from fsrc import glm

if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    print(train_dataset.tag_dict)


    myglm = glm.GlobalLinearModel(train_dataset, dev_dataset)
    # print(myglm.epsilon)
    # print(myglm.tag_dict)

    print(myglm.tag_dict)
    # exit()
    print(myglm.reversed_tag_dict)

    myglm.online_training(10)
