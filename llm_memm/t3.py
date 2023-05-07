from src import dataloader
from src import llm

a = [1, 2, 3]
b = ['a', 'b', 'c']

print(zip(a, b))

for i in zip(a, b):
    print(i)


if __name__ == '__main__':
    train_path = './src/train.conll'
    dev_path = './src/dev.conll'

    train_dataset = dataloader.DataLoader(train_path, batch_size=10)
    print(train_dataset)
    dev_dataset = dataloader.DataLoader(dev_path, batch_size=10)
    print(dev_dataset)
    # print(train_dataset.batch_data[0])
    # print(len(train_dataset.batch_data[0]))
    train_batch = train_dataset.batch_data
    dev_batch = dev_dataset.batch_data

    sent = train_dataset.sent_word_list[0]
    tag = train_dataset.sent_tag_list[0]
    print(sent)
    print(tag)

    myllm = llm.LogLinearModel(train_dataset, dev_dataset)
    print(myllm.ftpl_instantialize(sent, 0, 'NR'))
    all_fv = [myllm.ftpl_instantialize(sent, 0, t) for t in train_dataset.tag_dict]
    print(all_fv)
    print(len(all_fv))
    # myllm.create_feature_space()
    myllm.gradient_descent01(30)
