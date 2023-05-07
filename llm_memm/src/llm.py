import numpy as np
from scipy.special import logsumexp

class LogLinearModel():
    def __init__(self, dataset, devdataset):

        self.dataset = dataset
        self.dev = devdataset
        self.model = {}

        self.epsilon = self.create_feature_space()
        self.d = len(self.epsilon)

        self.W = np.zeros(self.d)
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
        print('---feature space constructing over---')
        return {v:k for k,v in enumerate(list(epsilon))}

    def seqlize(self):
        '''
                        for a feature vect (after instantialize),
                        convert this vect to a num-vect

                        given (X, i, y_i), instantialize it into a feature vector,
                        this will be the input.

                        for each element of this input feature vec, we assign the index of this element in the epsilon

                :return:
                    seqlized_vec: len(seqlized_vec) == len(feature_vector)
                    final_vec: len(final_vec) == len(epsilon)
                                [0, 0, ..., 1,..., 3, 0,...0]
                                *** note ***
                                it should be noted that this final_vec is not W, it is actually the F
        '''

    def dot(self, f):
        score = 0
        for i in f:
            if (i in self.model):
                score += self.model[i]
        return score

    def scorer(self, feature_vec):
        '''

        :param feature_vec:
                computing w·f(Sj,i,t)
        :return:

        W = [w1, w2, w3]

        given (sj, i, t), we instantialize it into a feature vec
        [f2, f3, f2]
        theoretically, we seqlize this feature vec into a num vec, given epsilon,
        [0, 2, 1]
        then we compute the score with dot product:
        [0, 2, 1] * [w1, w2, w3] = 2w2+w3
        but for oov problem, seqlize step is impractical
        [f2, f3, f2, f4, f5] = 2w2+w3
        thus we skip the seqlize step
            if fi in epsilon, we orderly find the corresponding wi in W, and sum all wi
            e.g. [f2, f3, f2] ---> [w2, w3, w2]
        '''
        score = 0
        for f in feature_vec:
            # print(f)
            if f in self.epsilon:
                # print(f)
                f_id = self.epsilon[f]
                # print(f_id)
                score += self.W[f_id]
        return score

    def predict(self, sent, position_id):
        maxnum = -1e10
        # tmp_score = 0
        tag = "NULL"
        for t in self.dataset.tag_dict:
            fv = self.ftpl_instantialize(sent, position_id, t)
            tmp_score = self.scorer(fv)
            if (tmp_score > (maxnum + 1e-10)):
                maxnum = tmp_score
                tag = t
        return tag

    def online_training(self, epochs):
        sent_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            for j in range(len(sent_list)):
                for i in range(len(sent_list[j])):
                    t_i = sent_tag_list[j][i]
                    predict_tag = self.predict(sent_list[j], i)
                    if predict_tag != t_i:
                        fv_correct = self.ftpl_instantialize(sent_list[j], i, t_i)
                        fv_predict = self.ftpl_instantialize(sent_list[j], i, predict_tag)

                        # updating W
                        for f_correct in fv_correct:
                            if f_correct in self.epsilon:
                                correct_id = self.epsilon[f_correct]
                                self.W[correct_id] += 1

                        for f_predict in fv_predict:
                            if f_predict in self.epsilon:
                                predict_id = self.epsilon[f_predict]
                                self.W[predict_id] -= 1

            print('---training set---')
            self.evaluate(self.dataset)
            print('---dev set---')
            self.evaluate(self.dev)
        print('---online training over---')

    def evaluate(self, dataset):
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
                max_tag = self.predict(sent_word_list[j], i)
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

        print(
            f'num correctly-predict sent: {num_correct_sent / num_sent}, acc over all tag: {num_correct_tag / num_tag}')


    # def build_batch(self, batch_size):
    #     num_batch = len()

    def gradient_descent01(self, epochs):
        '''
                this is an implementation of original version
        :param batch: len(batch) = batch_size, [([word], [tag]), ]
        :param learning_rate:
        :param lmbda: L2正则化系数
        :return:

        model define:
            given sent:x, and pos_index: i,
            P(t|x, i) = exp(score(x, i, t)) / sum_t' (exp(score(x, i, t')))
            score(x, i, t) = w * f(x, i, t)

        optimization objective:
        in training dataset,
        min NLL
        NLL = - sum_(x, i ,t) [logP(t|x,i,w)]
            = - sum_(x, i ,t) [log(exp(score(x, i, t))) - log(sum_t' (exp(x, i, t'))]

        梯度更新
        w_k+1 = w_k - eta * g

            g对w对偏导
        g = - sum_(x, i ,t) [
                     f(x, i, t): 正确的t
                    - sum_t' ( P(t'|x, i)*f(x, i, t') )
        ]



        '''
        B = 50
        b = 0
        lr = 0.01

        sent_word_list = self.dataset.sent_word_list
        sent_tag_list = self.dataset.sent_tag_list

        for m in range(epochs):
            for j in range(len(sent_word_list)):
                for i in range(len(sent_word_list[j])):
                    correct_tag = sent_tag_list[j][i]
                    # predict_tag = self.predict(sent_word_list[j], i)
                    fv_correct = self.ftpl_instantialize(sent_word_list[j], i, correct_tag)
                    for f in fv_correct:
                        if f in self.epsilon:
                            correct_id = self.epsilon[f]
                            self.g[correct_id] += 1
                    '''
                        NLL = - sum_(x, i ,t) [logP(t|x,i,w)]
                            = - sum_(x, i ,t) [
                                    log(exp(score(x, i, t))) - log(sum_t' (exp(x, i, t'))
                                    ]
                    '''
                    all_fv = [self.ftpl_instantialize(sent_word_list[j], i, t) for t in self.dataset.tag_dict]
                    all_socre = [self.scorer(fv) for fv in all_fv]
                    logsum_all = logsumexp(all_socre)

                    # exp_all_score = np.exp(all_socre)
                    # log_all_socre = np.log(exp_all_score)
                    # log_prob_list = np.array(log_all_socre) - logsum_all
                    log_prob_list = np.array(all_socre) - logsum_all

                    for fv_per_t_prime,prob_per_t_prime in zip(all_fv, log_prob_list):
                        for f in fv_per_t_prime:
                            if f in self.epsilon:
                                f_id = self.epsilon[f]
                                self.g[f_id] -= np.exp(prob_per_t_prime)

                    b += 1
                    if b == B:
                        self.W += lr * self.g
                        b = 0
                        self.g = np.zeros(self.d)

            '''
                if b > 0: means that there are remaining samples
            '''
            if b > 0:
                self.W += lr * self.g
                b = 0
                self.g = np.zeros(self.d)
            print('---training set---')
            self.evaluate(self.dataset)
            print('---dev set---')
            self.evaluate(self.dev)
        print('---online training over---')






