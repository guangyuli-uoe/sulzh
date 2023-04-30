
class LinearModel():
    def __init__(self, dataset, devdataset):
        self.W = {}
        self.dataset = dataset
        self.dev = devdataset
        self.model = {}

    def ftpl_instantialize(self, sent, i, t):
        '''

        :param sent: a list of word (after Chinese tokenization)
        :param i: index of word in current sent
        :param t: tag
        :return: feature vector
        '''

        features = []

        w_i = sent[i]
        w_i_pre = sent[i - 1] if i > 0 else '$'
        w_i_next = sent[i + 1] if i == len(sent) + 1 else '$$'

        '''
            c_(i,k): the k_th Chinese character of w_i
        '''

        features.append('02_' + t + '_' + w_i)
        features.append('03_' + t + '_' + w_i_pre)
        features.append('04_' + t + '_' + w_i_next)
        features.append('05_' + t + '_' + w_i + '_' + w_i_pre[-1])
        features.append('06_' + t + '_' + w_i + '_' + w_i_next[0])
        features.append('07_' + t + '_' + w_i[0])
        features.append('08_' + t + '_' + w_i[-1])

        # f09 = w_i[1:-1] if len(w_i) >= 3 else w_i
        # features.append('09_' + t + '_' + f09)
        # features.append('10_' + w_i[0] + '_' + f09)
        # features.append('11_' + w_i[-1] + '_' + f09)

        for k in range(1, len(w_i)-1):
            # print('09')
            features.append('09_' + t + '_' + w_i[k])
            features.append('10_' + t + '_' + w_i[0] + '_' + w_i[k])
            features.append('11_' + t + '_' + w_i[-1] + '_' + w_i[k])

        if len(w_i) == 1:
            features.append('12_' + t + '_' + w_i + '_' + w_i_pre[-1] + '_' + w_i_next[0])

        # if len(w_i) >= 2:
        #     flag = 0
        #     for k in range(0, len(w_i) - 1):
        #         if w_i[k] == w_i[k + 1]:
        #             flag = 1
        #         else:
        #             flag = 0
        #     if flag == 1:
        #         features.append('13_' + w_i[0] + '_' + 'consecutive')

        for k in range(len(w_i)):
            if k < len(w_i)-1:
                if w_i[k] == w_i[k + 1]:
                    features.append('13_' + t + '_' + w_i[k] + '_' + 'consecutive')

        for k in range(len(w_i) + 1):
            if len(w_i) >= 4:
                if 0 < k <= 4:
                    features.append('14_' + t + '_' + w_i[:k])
                    features.append('15_' + t + '_'+ w_i[-k:])
            else:
                if k > 0:
                    features.append('14_' + t + '_' + w_i[:k])
                    features.append('15_' + t + '_' + w_i[-k:])

        return features


    def create_feature_space(self):
        '''

        :return: whole feature space
        '''
        epsilon = set()

        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for sentid,sent in enumerate(sent_list):
            tag_list = sent_tag_list[sentid]
            for i,tag in enumerate(tag_list):
                tmp_features = self.ftpl_instantialize(sent, i, tag)
                for f in tmp_features:
                    epsilon.add(f)
                # print(tmp_features)
                # break
        for feature in list(epsilon):
            self.model[feature] = 0
        return list(epsilon)

    def dot(self, f):
        score = 0
        for i in f:
            if (i in self.model):
                score += self.model[i]
        return score

    def max_tag(self, sentence, pos):
        maxnum = -1e10
        tempnum = 0
        tag = "NULL"
        for t in self.dataset.tag_dict:
            fv = self.ftpl_instantialize(sentence, pos, t)
            tempnum = self.dot(fv)
            if (tempnum > (maxnum + 1e-10)):
                maxnum = tempnum
                tag = t
        return tag

    def online_training(self, epochs):
        max_train_precision = 0
        max_dev_precision = 0
        sent_word_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list
        for iterator in range(0, epochs):
            print("iterator " + str(iterator))
            wordCount = 0
            # for s in self.train.sentences:
            #     print(s)
            #     for p in range(0, len(s.word)):
            #         max_tag = self.max_tag(s, p)
            #         correcttag = s.tag[p]
            #         if (max_tag != correcttag):
            # for sent_word in self.dataset.sent_word_list:
            #     for i

            for j in range(len(sent_word_list)):
                for i in range(len(sent_word_list[j])):
                    max_tag = self.max_tag(sent_word_list[j], i)
                    correcttag = sent_tag_list[j][i]
                    if (max_tag != correcttag):
                        fmaxtag = self.ftpl_instantialize(sent_word_list[j], i, max_tag)
                        fcorrecttag = self.ftpl_instantialize(sent_word_list[j], i, correcttag)
                        for i in fmaxtag:
                            if (i in self.model):
                                self.model[i] -= 1
                        for i in fcorrecttag:
                            if (i in self.model):
                                self.model[i] += 1

            print('---training set---')
            self.evaluate(self.dataset, epochs)
            print('---dev set---')
            self.evaluate(self.dev, epochs)
        print('---online training over---')
        # print(self.model)

    def evaluate(self, dataset, epoch):
        sent_word_list = dataset.sent_word_list
        sent_tag_list = dataset.sent_tag_list

        num_sent = len(sent_word_list)
        num_tag = 0
        num_correct_tag = 0
        tmp_tag_list = []
        num_correct_sent = 0

        for j in range(len(sent_word_list)):
            gold_tag_list = sent_tag_list[j]

            for i in range(len(sent_word_list[j])):
                num_tag += 1
                max_tag = self.max_tag(sent_word_list[j], i)
                tmp_tag_list.append(max_tag)
                correcttag = sent_tag_list[j][i]
                # print(f' epoch: {epoch},sent_word_list[j]: {sent_word_list[j]}, {i}, correcttag: {correcttag}, max_tag: {max_tag}')
                if (max_tag != correcttag):
                    pass
                else:
                    num_correct_tag += 1
            # print(f'gold_tag_list: {gold_tag_list}')
            # print(f'tmp_tag_list: {tmp_tag_list}')

            if gold_tag_list == tmp_tag_list:

                num_correct_sent += 1
            tmp_tag_list = []

        print(f'num correctly-predict sent: {num_correct_sent/num_sent}, acc over all tag: {num_correct_tag/num_tag}')