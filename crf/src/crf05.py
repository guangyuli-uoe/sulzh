import numpy as np
# from scipy.misc import logsumexp
from scipy.special import logsumexp
# import torch

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
        self.N = len(self.tag_dict)

        self.tag_transition_matrix = self.buil_transition_matrix()

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
        # self.tag_transition_matrix = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        return {v: k for k, v in enumerate(list(epsilon))}

    def scorer(self, fv_list):
        score = 0
        for f in fv_list:
            if f in self.epsilon:
                f_id = self.epsilon[f]
                score += self.W[f_id]
        return score

    def buil_transition_matrix(self):
        # N = len(self.dataset.tag_dict)
        tag_transition_matrix = np.zeros((self.N, self.N))
        for i, t_i_1 in enumerate(self.tag_dict):
            for j, t_i in enumerate(self.tag_dict):
                a = self.create_tag_bigram_feature(t_i_1, t_i)
                score = self.scorer(a)
                tag_transition_matrix[i, j] = score
        return tag_transition_matrix

    def viterbi_predict(self, sentence):
        T = len(sentence)
        dp = np.zeros((T, self.N))
        path = np.zeros((T, self.N), dtype=int)

        init_fv = [self.create_feature_template(sentence, 0, self.BOS, tag)
                   for tag in self.dataset.tag_dict]
        dp[0] = np.array([self.scorer(fv_list) for fv_list in init_fv])
        # print(dp)
        emit_matrix = np.zeros((T, self.N))
        for t in range(1, T):
            for column,tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[t, column] = emit_score

        for t in range(1, T):
            scores = self.tag_transition_matrix + emit_matrix[t].reshape(1, -1) + dp[t-1].reshape(-1, 1)
            path[t] = np.argmax(scores, axis=0)
            dp[t] = np.max(scores, axis=0)
        predict_n = np.argmax(dp[-1])
        result = [predict_n]
        for i in reversed(range(1, T)):
            predict_n = path[i, predict_n]
            # print(result)
            result.append(predict_n)
        return [self.reversed_tag_dict[i] for i in reversed(result)]

    def forward(self, sentence):
        T = len(sentence)
        alpha = np.zeros((T, self.N))

        init_fv = [self.create_feature_template(sentence, 0, self.BOS, tag)
                   for tag in self.dataset.tag_dict]
        alpha[0] = np.array([self.scorer(fv_list) for fv_list in init_fv])

        emit_matrix = np.zeros((T, self.N))
        for t in range(1, T):
            for column, tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[t, column] = emit_score

        for t in range(1, T):
            alpha[t] = logsumexp(self.tag_transition_matrix + emit_matrix[t].reshape(1, -1) + alpha[t-1].reshape(-1, 1), axis=0)

        return alpha

    def backward(self, sentence):
        T = len(sentence)
        beta = np.zeros((T, self.N))
        beta[-1] = 0

        emit_matrix = np.zeros((T, self.N))
        for t in range(1, T):
            for column, tag in enumerate(self.tag_dict):
                fv = self.ftpl_instantialize(sentence, t, tag)
                emit_score = self.scorer(fv)
                emit_matrix[t, column] = emit_score

        for t in reversed(range(len(sentence) - 1)):
            # beta[t] = ().logsumexp(-1)
            beta[t] = logsumexp(self.tag_transition_matrix + emit_matrix[t+1].reshape(1, -1) + beta[t+1].reshape(1, -1), axis=-1)
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

        alpha = self.forward(sentence)
        beta = self.backward(sentence)
        # print(f'alpha.shape: {alpha.shape}')
        # print(f'beta.shape: {beta.shape}')
        # Z = logsumexp(alpha[:,-1])
        Z = logsumexp(alpha[-1])
        # print(Z)
        # exit()

        for i,tag in enumerate(self.tag_dict):
            features = self.create_feature_template(sentence, 0, self.BOS, tag)
            features_id = [self.epsilon[f] for f in features if f in self.epsilon]
            # p = torch.exp(self.scorer(features) + beta[i, 0] - Z)
            p = np.exp(self.scorer(features) + beta[0, i] - Z)
            # print(p)
            # exit()
            for id in features_id:
                self.g[id] -= p


        for t in range(1, len(sentence)):
            # print(len(sentence))
            for i,cur_tag in enumerate(self.tag_dict):
                for j,pre_tag in enumerate(self.tag_dict):
                    emit_features = self.ftpl_instantialize(sentence, t, cur_tag)
                    emit_fids = [self.epsilon[f] for f in emit_features if f in self.epsilon]
                    emit_score = self.scorer(emit_features)

                    transition_features = self.create_tag_bigram_feature(pre_tag, cur_tag)
                    transition_fids = [self.epsilon[f] for f in transition_features if f in self.epsilon]
                    transition_score = self.scorer(transition_features)

                    score = transition_score + emit_score
                    # print(score)
                    # exit()
                    '''
                        i: current_tag
                        j: previous_tag
                        t: 
                    '''
                    # p = torch.exp(alpha[j, t-1] + score + beta[i, t] - Z)
                    p = np.exp(alpha[t - 1, j] + score + beta[t, i] - Z)
                    # print(p)
                    # print(len(p))
                    # exit()

                    for id in emit_fids:
                        self.g[id] -= p
                    for id in transition_fids:
                        self.g[id] -= p

    def sgd_training(self):
        # lr = 0.3
        lr = 0.5
        # batch_size = 10
        batch_size = 1
        epochs = 50
        b = 0

        sent_word_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            # print(f'current in epoch: {m}')
            for i in range(len(sent_word_list)):
                # print(f'epoch: {m}, sample: {i}')
                b += 1
                sentence = sent_word_list[i]
                tags = sent_tag_list[i]
                self.update_gradient(sentence, tags)
                if b == batch_size:
                    self.W += lr * self.g
                    b = 0
                    self.g = np.zeros(self.d)
                    self.tag_transition_matrix = self.buil_transition_matrix()

            if b > 0:
                self.W += lr * self.g
                b = 0
                self.g = np.zeros(self.d)
                self.tag_transition_matrix = self.buil_transition_matrix()

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













