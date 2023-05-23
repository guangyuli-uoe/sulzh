import numpy as np

class GlobalLinearModel():
    def __init__(self,dataset, devdataset):
        self.BOS = 'BOS'
        self.dataset = dataset
        self.dev = devdataset
        self.epsilon = self.create_feature_space()
        self.d = len(self.epsilon)
        self.W = np.zeros(self.d)
        # self.tag_dict = {v: k for k, v in enumerate([self.BOS] + list(self.dataset.tag_dict.keys()))}

        self.tag_dict = self.dataset.tag_dict
        self.reversed_tag_dict = {v: k for k, v in self.tag_dict.items()}

        self.bigram_features = [
            [self.create_tag_bigram_feature(prev_tag, tag) for prev_tag in self.dataset.tag_dict]
            for tag in self.dataset.tag_dict
        ]

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

    def create_tag_bigram_feature(self, pre_tag, cur_tag):
        return ['01:' + cur_tag + '_' + pre_tag]

    def create_feature_template(self, sentence, position, pre_tag, cur_tag):
        template = []
        template.extend(self.create_tag_bigram_feature(pre_tag, cur_tag))
        template.extend(self.ftpl_instantialize(sentence, position, cur_tag))
        return template

    def create_feature_space(self):
        epsilon = set()

        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for j,sent in enumerate(sent_list):
            tag_list = sent_tag_list[j]
            for i in range(len(tag_list)):
                if i == 0:
                    tag_pre = self.BOS
                else:
                    tag_pre = tag_list[i-1]
                tmp_fv = self.create_feature_template(sent_list[j], i, tag_pre, tag_list[i])
                for f in tmp_fv:
                    epsilon.add(f)
        print('---feature space constructing over---')
        return {v: k for k, v in enumerate(list(epsilon))}

    def scorer(self, fv):
        score = 0
        for f in fv:
            if f in self.epsilon:
                f_id = self.epsilon[f]
                score += self.W[f_id]
        return score

    def viterbi_predict(self, sentence):
        T = len(sentence)
        N = len(self.dataset.tag_dict)

        dp = np.zeros((N, T))
        path = np.zeros((N, T))

        '''
            transition matrix
            i-->j
        '''

        tag_transition_matrix = np.zeros((N, N))
        for i,t_i_1 in enumerate(self.tag_dict):
            for j,t_i in enumerate(self.tag_dict):
                a = self.create_tag_bigram_feature(t_i_1, t_i)
                score = self.scorer(a)
                tag_transition_matrix[i, j] = score

        '''
            initialization
            填第0列
        '''
        init_fv = [self.create_feature_template(sentence, 0, self.BOS, tag)
               for tag in self.dataset.tag_dict]

        dp[:,0] = [self.scorer(fv) for fv in init_fv]

        '''
            recursion
            填1到T-1列
        '''
        # for i in range(1, T): # 列
        #
        #     for j in range(N): # 行，current tag
        #         current_tag = self.reversed_tag_dict[j]
        #         candidates = []
        #         # for j in range(N):
        #         '''
        #             算 dp[j, i]
        #         '''
        #         for k,tag_i in enumerate(self.tag_dict): # tag_i -> j
        #             # emit_fv = self.create_feature_template(sentence,)
        #             # emit_score =
        #             # tmp_score = tag_transition_matrix[k,j] +
        #             transition_score = tag_transition_matrix[k,j]
        #             emit_fv = self.create_feature_template(sentence, j, tag_i, current_tag)
        #             emit_score = self.scorer(emit_fv)
        #             final_score = transition_score + emit_score + dp[k, i-1]

        for t in range(1, T):

            for s in range(N):
                current_tag = self.reversed_tag_dict[s]
                candidates = []
                for s_ in range(N):
                    # previous_tag = self.reversed_tag_dict[s_]
                    transition_score = tag_transition_matrix[s_, s]
                    emit_fv = self.ftpl_instantialize(sentence, t, current_tag)
                    emit_score = self.scorer(emit_fv)
                    final_score = transition_score + emit_score + dp[s_, t - 1]
                    candidates.append(final_score)
                dp[s, t] = np.max(candidates)
                path[s, t] = np.argmax(candidates)

        predict_n = np.argmax(dp[:,-1])
        result = [predict_n]
        for i in reversed(range(1, T)):
            prev = path[predict_n, i]
            result.append(prev)
        return [self.reversed_tag_dict[i] for i in reversed(result)]

    def viterbi_predict11(self, sentence):
        T = len(sentence)
        N = len(self.dataset.tag_dict)

        dp = np.zeros((N, T))
        path = np.zeros((N, T))

        '''
            transition matrix
            i-->j
        '''
        self.bigram_features = [
            [self.create_tag_bigram_feature(prev_tag, tag) for prev_tag in self.dataset.tag_dict]
            for tag in self.dataset.tag_dict
        ]
        tag_transition_matrix = np.zeros((N, N))
        for i, t_i_1 in enumerate(self.tag_dict):
            for j, t_i in enumerate(self.tag_dict):
                a = self.create_tag_bigram_feature(t_i_1, t_i)
                score = self.scorer(a)
                tag_transition_matrix[i, j] = score
        '''
            initialization
            填第0列
        '''
        init_fv = [self.create_feature_template(sentence, 0, self.BOS, tag)
                   for tag in self.dataset.tag_dict]
        dp[:, 0] = [self.scorer(fv) for fv in init_fv]

        '''
            emition matrix
        '''
        emit_matrix = np.zeros((N, T))
        for t in range(1, T):
            for row,tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[row,t] = emit_score
        '''
            recursion
            1-T
        '''
        for t in range(1, T):
            # emit_score = np.array([
            #     self.scorer(self.ftpl_instantialize(sentence, t, tag))
            #     for tag in self.dataset.tag_dict
            # ])
            # emit_score = emit_matrix[:,t].reshape(-1, 1)# （31， 1）
            emit_score = emit_matrix[:, t].reshape(1, -1) # (1, 31)

            scores = tag_transition_matrix + emit_score + dp[:,t-1, None]
            # dp[:, t-1].reshape(-1, 1)
            # print(scores.shape)
            # exit()
            path[:,t] = np.argmax(scores, axis=0)
            dp[:,t] = np.max(scores, axis=0)

        predict_n = np.argmax(dp[:, -1])
        # print(predict_n)
        result = [predict_n]
        for i in reversed(range(1, T)):
            predict_n = path[predict_n, i]
            # print(result)
            result.append(predict_n)
        return [self.reversed_tag_dict[i] for i in reversed(result)]









    def viterbi_predict02(self, sentence):

        newsentence = ['$'] + sentence
        T = len(newsentence)
        # T = len(sentence)
        N = len(self.tag_dict)

        # dp = np.zeros((N, T))
        # path = np.zeros((N, T))
        tb = np.zeros((N, T))
        bp = np.zeros((N, T), dtype='int')  # path

        tb[0, 0] = 0
        '''
            sentence: 1-n
            newsentence: 0-n
        '''
        for i in range(1, T): # 1, n
            for t in range(1, N): # current tag
                for t_prime in range(N): # previous tag
                    fv = self.create_feature_template(sentence, i, self.reversed_tag_dict[t_prime], self.reversed_tag_dict[t])
                    score = self.scorer(fv)
                    if tb[t_prime, i-1] + score > tb[t, i]:
                        tb[t, i] = tb[t_prime, i-1] + score
                        bp[t, i] = t_prime

        y_n = np.argmax(tb[:,-1])
        output = [y_n]
        for i in reversed(range(2, T)):
            y = bp[y_n, i]
            output.append(y)

        # reversed_tag_dict = {v: k for k, v in self.tag_dict.items()}
        # print(reversed_tag_dict)
        return [self.reversed_tag_dict[i] for i in reversed(output)]

    def viterbi_predict111(self, sentence):
        T = len(sentence)
        delta = np.zeros((T, len(self.dataset.tag_dict)))
        paths = np.zeros((T, len(self.dataset.tag_dict)), dtype='int')

        bigram_scores = np.array([
            [self.scorer(bifv) for bifv in bifvs]
            for bifvs in self.bigram_features
        ])
        # print(bigram_scores.shape) # (31, 31)
        # print(bigram_scores)
        fvs = [self.create_feature_template(sentence, 0, self.BOS, tag)
               for tag in self.dataset.tag_dict]
        delta[0] = [self.scorer(fv) for fv in fvs]
        # print(f'fvs: {fvs}') # (31, num of f)
        # print(f'len(fvs): {len(fvs)}') # len(fvs): 31
        # print(delta[0])

        for i in range(1, T):
            unigram_scores = np.array([
                self.scorer(self.ftpl_instantialize(sentence, i, tag))
                for tag in self.dataset.tag_dict
            ])
            # print(f'unigram_scores.shape: {unigram_scores.shape}') # unigram_scores.shape: (31,)
            # print(f'unigram_scores[:, None]: {unigram_scores[:, None].shape}') # unigram_scores[:, None]: (31, 1)
            scores = bigram_scores + unigram_scores[:, None] + delta[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            # print(f'paths[i]: {paths[i]}')
            delta[i] = scores[np.arange(len(self.dataset.tag_dict)), paths[i]]
            # print(f'delta[i]: {delta[i]}')
            # break
        prev = np.argmax(delta[-1])
        # print(f'prev: {prev}')
        predict = [prev]
        for i in reversed(range(1, T)):
            prev = paths[i, prev]
            predict.append(prev)
        # print(predict)
        # exit()
        # v: k for k, v
        reversed_tag_dict = {v: k for k,v in self.dataset.tag_dict.items()}
        # print(reversed_tag_dict)
        return [reversed_tag_dict[i] for i in reversed(predict)]


    def online_training(self, epochs):
        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            # print(m)
            for j in range(len(sent_list)):
                # print(j)
                sentence = sent_list[j]
                tags = sent_tag_list[j]
                predict = self.viterbi_predict111(sentence)
                # print(f'p: {predict}')
                # print(f't: {tags}')

                if predict != tags:
                    for i in range(len(tags)):

                        if i == 0:
                            gold_pre_tag = self.BOS
                            predict_pre_tag = self.BOS
                        else:
                            gold_pre_tag = tags[i - 1]
                            predict_pre_tag = predict[i - 1]
                        gold_feature = self.create_feature_template(sentence, i, gold_pre_tag, tags[i])
                        predict_feature = self.create_feature_template(sentence, i, predict_pre_tag, predict[i])
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
            predict_tags = self.viterbi_predict111(sent_word_list[j])
            # print(gold_tag_list)
            # print(predict_tags)
            # exit()
            total_tag_num += len(gold_tag_list)
            for i in range(len(sent_word_list[j])):
                if predict_tags[i] == gold_tag_list[i]:
                    correct_tag_num += 1
        print(f'acc: {correct_tag_num/total_tag_num}')





