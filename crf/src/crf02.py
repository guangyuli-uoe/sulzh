import numpy as np
# from scipy.misc import logsumexp
from scipy.special import logsumexp
import torch

class LinearChainCRF():

    def __init__(self, dataset, devdataset):
        self.BOS = 'BOS'
        self.dataset = dataset
        self.dev = devdataset
        self.tag_dict = self.dataset.tag_dict
        self.reversed_tag_dict = {v: k for k, v in self.tag_dict.items()}

        self.epsilon = self.create_feature_space()
        self.d = len(self.epsilon)
        self.W = np.zeros(self.d)

        self.tag_transition_matrix = self.buil_transition_matrix()
        # self.bigram_features = [
        #     [self.create_tag_bigram_feature(prev_tag, tag) for prev_tag in self.dataset.tag_dict]
        #     for tag in self.dataset.tag_dict
        # ]
        # self.bigram_scores = np.array([
        #     [self.scorer(bifv) for bifv in bifvs]
        #     for bifvs in self.bigram_features
        # ])

        self.g = np.zeros(self.d)

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

    def buil_transition_matrix(self):
        N = len(self.dataset.tag_dict)
        tag_transition_matrix = np.zeros((N, N))
        for i, t_i_1 in enumerate(self.tag_dict):
            for j, t_i in enumerate(self.tag_dict):
                a = self.create_tag_bigram_feature(t_i_1, t_i)
                score = self.scorer(a)
                tag_transition_matrix[i, j] = score
        return tag_transition_matrix

    def viterbi_predict(self, sentence):
        T = len(sentence)
        N = len(self.dataset.tag_dict)
        dp = np.zeros((N, T))
        path = np.zeros((N, T),dtype='int')
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
            emit_score = emit_matrix[:, t].reshape(1, -1) # (1, 31)
            scores = self.tag_transition_matrix + emit_score + dp[:, t-1].reshape(-1, 1)
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

    def forward(self, sentence):
        T = len(sentence)
        N = len(self.dataset.tag_dict)

        alpha = np.zeros((N, T))
        '''
                    initialization
                    填第0列
        '''
        init_fv = [self.create_feature_template(sentence, 0, self.BOS, tag)
                   for tag in self.dataset.tag_dict]
        alpha[:, 0] = [self.scorer(fv) for fv in init_fv]

        '''
                    emition matrix
        '''
        emit_matrix = np.zeros((N, T))
        for t in range(1, T):
            for row, tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[row, t] = emit_score

        for t in range(1, T):
            emit_score = emit_matrix[:, t].reshape(1, -1)  # (1, 31)
            scores = self.tag_transition_matrix + emit_score + alpha[:, t - 1].reshape(-1, 1)
            alpha[:, t] = logsumexp(scores, axis=0)

        return alpha

    def backward(self, sentence):
        T = len(sentence)
        N = len(self.dataset.tag_dict)

        beta = np.zeros((N, T))

        emit_matrix = np.zeros((N, T))
        for t in range(1, T):
            for row, tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[row, t] = emit_score
        for t in reversed(range(len(sentence) - 1)):
            emit_score = emit_matrix[:, t+1].reshape(1, -1)# (1, 31)
            scores = self.tag_transition_matrix + emit_score + beta[:, t+1].reshape(1, -1)
            beta[:, t] = logsumexp(scores, axis=1)
        return beta

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
            else:
                pre_tag = tags[i-1]
            current_tag = tags[i]
            feature = self.create_feature_template(sentence, i, pre_tag, current_tag)
            for f in feature:
                if f in self.epsilon:
                    fid = self.epsilon[f]
                    self.g[fid] += 1

        # logZ = alpha[:, -1].logsumexp(0)
        alpha = self.forward(sentence)
        beta = self.backward(sentence)
        Z = logsumexp(alpha[:, -1], axis=0)

        for i,tag in enumerate(self.tag_dict):
            features = self.create_feature_template(sentence, 0, self.BOS, tag)
            features_id = [self.epsilon[f] for f in features if f in self.epsilon]
            p = np.exp(self.scorer(features) + beta[i, 0] - Z)
            # p = np.exp(self.scorer(features) + beta[0, i] - Z)
            for id in features_id:
                self.g[id] -= p

        for i in range(1, len(sentence)):
            for j, tag in enumerate(self.tag_dict):
                pass