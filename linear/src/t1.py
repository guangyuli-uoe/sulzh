from hmm_taggin.newsrc import dataloader

if __name__ == '__main__':
    training_path = '../../hmm_taggin/train.conll'
    dataset = dataloader.Loader(training_path)

    # print(dataset.sent_list)
    print(dataset.align_dict)

    print('我是你爹'[-1])
    print(len('我是你爹i love you'))
    print(len('i love you'))

    print('我爱'[1:-1])

    # for char in '我爱我爱你你'[1: -1]:
    #     print(('09', 't', char))

    w_i = '我我爱'

    f09 = w_i[1:-1] if len(w_i) >= 3 else w_i
    print(f09)
    f10 = w_i[1:]
    print(f10)

    print('-----13-----')
    for i in range(0, len(w_i)-1):
        print(w_i[i])
        print(w_i[i+1])
        print('---')

    print('---13---')
    if len('1') == 1:
        flag = 0
        for k in range(0, len(w_i)-1):
            if w_i[k] == w_i[k+1]:
                flag = 1
            else:
                flag = 0

        if flag == 1:
            print('ok')
    print('-----9')
    a = '我爱你mm'
    for i in range(1, len(a)-1):
        print(a[i])
    print('---9-1---')
    for c in a[1:-1]:
        print(c)

    features = []
    for k in range(1, len(a) - 2):
        print(k)
        features.append('09_' + "t" + '_' + a[k])
        # features.append('10_' + "t" + '_' + a[0] + '_' + a[k])
        # features.append('11_' + "t" + '_' + a[-1] + '_' + a[k])
    print(features)
