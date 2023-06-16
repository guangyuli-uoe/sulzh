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

        # self.tag_transition_matrix = self.buil_transition_matrix()
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
        self.bigram_features = [
            [self.create_tag_bigram_feature(prev_tag, tag) for prev_tag in self.tag_dict]
            for tag in self.tag_dict
        ]
        self.bigram_scores = np.zeros((len(self.tag_dict), len(self.tag_dict)))
        return {v: k for k, v in enumerate(list(epsilon))}

    def scorer(self, fv):
        score = 0
        for f in fv:
            if f in self.epsilon:
                f_id = self.epsilon[f]
                score += self.W[f_id]
        return score

    def predict(self, sentence):
        states = len(sentence)
        type = len(self.tag_dict)

        max_score = np.zeros((states, type))
        paths = np.zeros((states, type), dtype='int')

        for j in range(type):
            feature = self.create_feature_template(sentence, 0, self.BOS, self.reversed_tag_dict[j])
            max_score[0][j] = self.scorer(feature)
            paths[0][j] = -1

        # 动态规划
        for i in range(1, states):
            unigram_scores = np.array([self.scorer(self.ftpl_instantialize(sentence, i, tag)) for tag in self.tag_dict])
            scores = self.bigram_scores + unigram_scores[:, None] + max_score[i - 1]
            paths[i] = np.argmax(scores, axis=1)
            max_score[i] = np.max(scores, axis=1)
            # for j in range(type):
            #     unigram_scores = self.score(self.create_unigram_feature(sentence, i, self.tags[j]))
            #     scores = unigram_scores + np.array(self.bigram_scores[j])
            #     max_score[i][j] = max(scores + max_score[i - 1])
            #     paths[i][j] = np.argmax(scores + max_score[i - 1])

        gold_path = []
        cur_state = states - 1
        step = np.argmax(max_score[cur_state])
        gold_path.insert(0, self.reversed_tag_dict[step])
        while True:
            step = int(paths[cur_state][step])
            if step == -1:
                break
            gold_path.insert(0, self.reversed_tag_dict[step])
            cur_state -= 1

        return gold_path



    def forward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tag_dict)))
        path_scores[0] = [self.scorer(self.create_feature_template(sentence, 0, self.BOS, tag))
                          for tag in self.tag_dict]
        for i in range(1, len(sentence)):
            unigram_scores = np.array([self.scorer(self.ftpl_instantialize(sentence, i, tag)) for tag in self.tag_dict])
            scores = self.bigram_scores + unigram_scores[:, None]
            path_scores[i] = logsumexp(path_scores[i - 1] + scores, axis=1)

        return path_scores

    def backward(self, sentence):
        path_scores = np.zeros((len(sentence), len(self.tag_dict)))

        for i in reversed(range(len(sentence) - 1)):
            unigram_scores = np.array(
                [self.scorer(self.ftpl_instantialize(sentence, i + 1, tag)) for tag in self.tag_dict])
            scores = self.bigram_scores.T + unigram_scores
            path_scores[i] = logsumexp(path_scores[i + 1] + scores, axis=1)
        return path_scores

    def update_gradient(self, sentence, tags):
        for i in range(len(sentence)):
            if i == 0:
                pre_tag = self.BOS
            else:
                pre_tag = tags[i - 1]
            cur_tag = tags[i]
            feature = self.create_feature_template(sentence, i, pre_tag, cur_tag)
            for f in feature:
                if f in self.epsilon:
                    self.g[self.epsilon[f]] += 1

        forward_scores = self.forward(sentence)
        backward_scores = self.backward(sentence)
        dinominator = logsumexp(forward_scores[-1])

        for i, tag in enumerate(self.tag_dict):
            features = self.create_feature_template(sentence, 0, self.BOS, tag)
            features_id = (self.epsilon[f] for f in features if f in self.epsilon)
            p = np.exp(self.scorer(features) + backward_scores[0, i] - dinominator)
            for id in features_id:
                self.g[id] -= p

        for i in range(1, len(sentence)):
            for j, tag in enumerate(self.tag_dict):
                unigram_features = self.ftpl_instantialize(sentence, i, tag)
                unigram_features_id = [self.epsilon[f] for f in unigram_features if f in self.epsilon]
                scores = self.bigram_scores[j] + self.scorer(unigram_features)
                probs = np.exp(scores + forward_scores[i - 1] + backward_scores[i, j] - dinominator)

                for bigram_feature, p in zip(self.bigram_features[j], probs):
                    bigram_feature_id = [self.epsilon[f]
                                         for f in bigram_feature if f in self.epsilon]
                    for fi in bigram_feature_id + unigram_features_id:
                        self.g[fi] -= p


    def sgd_training(self):
        lr = 0.3
        batch_size = 10
        epochs = 50
        b = 0

        sent_word_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            for i in range(len(sent_word_list)):
                b += 1
                sentence = sent_word_list[i]
                tags = sent_tag_list[i]
                self.update_gradient(sentence, tags)
                if b == batch_size:
                    self.W += lr * self.g
                    b = 0
                    self.g = np.zeros(self.d)
                    # self.tag_transition_matrix = self.buil_transition_matrix()
                    self.bigram_scores = np.array([
                        [self.scorer(f) for f in bigram_features]
                        for bigram_features in self.bigram_features
                    ])

                if b > 0:
                    self.W += lr * self.g
                    b = 0
                    self.g = np.zeros(self.d)
                    # self.tag_transition_matrix = self.buil_transition_matrix()
                    self.bigram_scores = np.array([
                        [self.scorer(f) for f in bigram_features]
                        for bigram_features in self.bigram_features
                    ])
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
            predict_tags = self.predict(sent_word_list[j])

            print(f'gold_tag_list: {gold_tag_list}')
            print(f'predict_tags: {predict_tags}')
            # exit()
            total_tag_num += len(gold_tag_list)
            for i in range(len(sent_word_list[j])):
                if predict_tags[i] == gold_tag_list[i]:
                    correct_tag_num += 1
        print(f'acc: {correct_tag_num/total_tag_num}')

