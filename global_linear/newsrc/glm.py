import numpy as np

class GlobalLinearModel():
    def __init__(self,dataset, devdataset):
        self.BOS = 'BOS'
        self.dataset = dataset
        self.dev = devdataset
        self.epsilon = self.create_feature_space()
        self.d = len(self.epsilon)

        self.W = np.zeros(self.d)

        self.tag_dict = {v:k for k,v in enumerate(['BOS']+list(self.dataset.tag_dict.keys()))}

    def ftpl_instantialize(self, sent, i, tags):
        features = []

        t_i = tags[i]
        t_i_pre = tags[i-1] if i != 0 else self.BOS

        w_i = sent[i]
        w_i_pre = sent[i - 1] if i > 0 else '$'
        w_i_next = sent[i + 1] if i == len(sent) + 1 else '$$'

        features.append('01_' + t_i + '_' + t_i_pre)

        features.append('02_' + t_i + '_' + w_i)
        features.append('03_' + t_i + '_' + w_i_pre)
        features.append('04_' + t_i + '_' + w_i_next)
        features.append('05_' + t_i + '_' + w_i + '_' + w_i_pre[-1])
        features.append('06_' + t_i + '_' + w_i + '_' + w_i_next[0])
        features.append('07_' + t_i + '_' + w_i[0])
        features.append('08_' + t_i + '_' + w_i[-1])

        for k in range(1, len(w_i)-1):
            # print('09')
            features.append('09_' + t_i + '_' + w_i[k])
            features.append('10_' + t_i + '_' + w_i[0] + '_' + w_i[k])
            features.append('11_' + t_i + '_' + w_i[-1] + '_' + w_i[k])

        if len(w_i) == 1:
            features.append('12_' + t_i + '_' + w_i + '_' + w_i_pre[-1] + '_' + w_i_next[0])

        for k in range(len(w_i)):
            if k < len(w_i)-1:
                if w_i[k] == w_i[k + 1]:
                    features.append('13_' + t_i + '_' + w_i[k] + '_' + 'consecutive')

        for k in range(len(w_i) + 1):
            if len(w_i) >= 4:
                if 0 < k <= 4:
                    features.append('14_' + t_i + '_' + w_i[:k])
                    features.append('15_' + t_i + '_'+ w_i[-k:])
            else:
                if k > 0:
                    features.append('14_' + t_i + '_' + w_i[:k])
                    features.append('15_' + t_i + '_' + w_i[-k:])

        return features

    def create_feature_space(self):
        epsilon = set()

        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for j,sent in enumerate(sent_list):
            tag_list = sent_tag_list[j]
            for i in range(len(tag_list)):
                tmp_fv = self.ftpl_instantialize(sent, i, tag_list)
                # print(tmp_fv)
                # exit()
                for f in tmp_fv:
                    epsilon.add(f)
        print('---feature space constructing over---')
        return {v: k for k, v in enumerate(list(epsilon))}

    def feature_aggregate(self, sentence, tags):
        '''
        :param sentence:['戴相龙', '说', '中国', '经济', '发展', '为', '亚洲', '作出', '积极', '贡献']
        :param tags: ['NR', 'VV', 'NR', 'NN', 'NN', 'P', 'NR', 'VV', 'JJ', 'NN']
        :return: sum of all sub-feature-instance of a sentence
        '''
        # len(sentence) == len(tags)
        aggregated_fv = []
        for i in range(len(sentence)):
            tmp_fv = self.ftpl_instantialize(sentence, i, tags)
            for f in tmp_fv:
                aggregated_fv.append(f)
        return aggregated_fv

    def global_scorer(self, aggregated_fv):
        '''

        :param aggregated_fv: sum of all sub-feature-instance of a sentence
        :return:
        '''
        score = 0
        for f in aggregated_fv:
            if f in self.epsilon:
                f_id = self.epsilon[f]
                score += self.W[f_id]
        return score

    def viterbi_predict(self, sentence, tags):
        T = len(['$'] + sentence)
        N = len(self.tag_dict) # bos + t

        tb = np.zeros((N, T))
        bp = np.zeros((N, T), dtype= int) # path

        aggregated_fv = self.feature_aggregate(sentence, tags)

        tb[0, 0] = 0
        for i in range(1, T):
            for t in range(1, N-1):
                for t_prime in range(N):
                    tmp_score = self.global_scorer(aggregated_fv)
                    if tb[t_prime, i-1] + tmp_score > tb[t, i]:
                        tb[t, i] = tb[t_prime, i-1] + tmp_score
                        bp[t, i] = t_prime
        # y_n = np.argmax(tb[:,-1])
        # print(y_n)
        # # exit()
        # predict = [y_n]

        prev = np.argmax(tb[:,-1])
        # print(f'prev: {prev}')
        predict = [prev]
        for i in reversed(range(1, T)):
            prev = bp[i, prev]
            predict.append(prev)
        # print(predict)
        # exit()
        # v: k for k, v
        reversed_tag_dict = {v: k for k, v in self.dataset.tag_dict.items()}
        # print(reversed_tag_dict)
        return [reversed_tag_dict[i] for i in reversed(predict)]


    def online_training(self, epochs):
        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            for j in range(len(sent_list)):
                sentence = sent_list[j]
                tags = sent_tag_list[j]
                predict = self.viterbi_predict(sentence, tags)

                if predict != tags:
                    for i in range(len(tags)):

                        if i == 0:
                            gold_pre_tag = self.BOS
                            predict_pre_tag = self.BOS
                        else:
                            gold_pre_tag = tags[i - 1]
                            predict_pre_tag = predict[i - 1]
                        gold_feature = self.ftpl_instantialize(sentence, i, gold_pre_tag, tags)
                        predict_feature = self.ftpl_instantialize(sentence, i, predict_pre_tag, predict)
                        for f in gold_feature:
                            if f in self.epsilon:
                                findex = self.epsilon[f]
                                self.W[findex] += 1
                        for f in predict_feature:
                            if f in self.epsilon:
                                findex = self.epsilon[f]
                                self.W[findex] -= 1

            print('---training set---')
            self.evaluate(self.dataset)
            print('---dev set---')
            self.evaluate(self.dev)
        print('---online training over---')

    def evaluate(self, dataset):
        sent_word_list = dataset.sent_word_list
        sent_tag_list = dataset.sent_tag_list

        total_tag_num = 0
        correct_tag_num = 0
        for j in range(len(sent_word_list)):
            gold_tag_list = sent_tag_list[j]
            predict_tags = self.viterbi_predict(sent_word_list[j])
            # print(gold_tag_list)
            # print(predict_tags)
            # exit()
            total_tag_num += len(gold_tag_list)
            for i in range(len(sent_word_list[j])):
                if predict_tags[i] == gold_tag_list[i]:
                    correct_tag_num += 1
        print(f'acc: {correct_tag_num/total_tag_num}')



