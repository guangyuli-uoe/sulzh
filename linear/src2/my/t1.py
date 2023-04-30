from linear.src2.lm import linear_model

if __name__ == '__main__':
    lm = linear_model()


    train_sentences = lm.train.sentences
    print(len(train_sentences))
    print(train_sentences[0].word)
    print(train_sentences[0].tag)
    print(train_sentences[0].wordchars)
    '''
    ['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
    ['NR', 'VV', 'NR', 'NN', 'NN', 'P', 'NR', 'VV', 'JJ', 'NN']
    [['戴', '相', '龙'], ['说'], ['中', '国'], ['经', '济'], ['发', '展'], ['为'], ['亚', '洲'], ['作', '出'], ['积', '极'], ['贡', '献']]
    '''

    lm.create_feature_space()
    print(lm.tags)
    print(len(lm.tags))
    '''
        {'NR': 1240, 'VV': 2808, 'NN': 5635, 'P': 779, 'JJ': 505, 'NT': 413, 'PU': 3016, 'OD': 86, 'M': 588, 'DEG': 562, 'CD': 611, 'LC': 322, 'CC': 275, 'AS': 181, 'PN': 280, 'AD': 1483, 'VA': 289, 'DEC': 484, 'ETC': 44, 'DT': 288, 'VC': 203, 'BA': 23, 'SB': 18, 'VE': 103, 'DEV': 30, 'LB': 9, 'CS': 30, 'MSP': 70, 'SP': 32, 'DER': 10, 'FW': 6}
        31
    '''
    # print(lm.model)
    a = []
    for v in lm.model.values():
        if v == 0:
            a.append(v)
    print(len(a))
    print(len(lm.model))

