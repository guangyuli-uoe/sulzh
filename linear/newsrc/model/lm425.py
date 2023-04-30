import numpy as np
from collections import Counter

class linearModel:

    '''
        aim: given X, ouput Y

        lm: score(X, i, y_i) = W * F(X, i, y_i)

        affine
        biaffine

        e.g.
            given a training set
                10000 sentences
                avg length per sent: 20

            1. instantialize and create feature space
            so you will get 20w instances (each instance will be like: (X, i, y_i))
            for each instance, we instantialize it according to a set of feature templates.
            thus you will get a final set of feature on this training data,
            we call this final set feature space, i.e. epsilon

            2. sequentialize
            given epsilon, for each element, we can assign a unique num in range of [0, len(epsilon)-1]




    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.epsilon = self.create_feature_space()
        self.d = len(self.epsilon)
        # for online training
        self.W = np.zeros(self.d)
        self.V = np.zeros(self.d)
        self.tag_list = dataset.tag_list

        # self.model = {}






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

        sent_list = self.dataset.sent_list
        sent_tag_list = self.dataset.sent_tag_list

        for sentid,sent in enumerate(sent_list):
            tag_list = sent_tag_list[sentid]
            for i,tag in enumerate(tag_list):
                tmp_features = self.ftpl_instantialize(sent, i, tag)
                for f in tmp_features:
                    epsilon.add(f)
                # print(tmp_features)
                # break

        return list(epsilon)

    def sequentialize(self, feature_vector):
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

        seqlized_vec = []
        final_vec = np.zeros(self.d)
        # count = Counter()
        # fre_dict = {}
        for feature in feature_vector:
            if feature in self.epsilon:
                seqlized_vec.append(self.epsilon.index(feature))
            # else:
            #     print('feature not included')
            #     print(feature)
        fre_dict = Counter(seqlized_vec)
        for id in seqlized_vec:
            final_vec[id] = fre_dict[id]

        # return seqlized_vec,final_vec,fre_dict
        return final_vec

    def online_training(self, epochs):
        sent_list = self.dataset.sent_list
        sent_tag_list = self.dataset.sent_tag_list

        k = 0
        # W = np.zeros(self.d)
        V = np.zeros(self.d)

        for m in range(epochs):
            for j in range(len(sent_list)):
                # print(j)
                for i in range(len(sent_list[j])):
                    print(i)
                    t_i = sent_tag_list[j][i] # current gold tag
                    # tmp_features = self.ftpl_instantialize(sent_list[j], i, t)
                    predict_tag = self.predict(sent_list[j], i)
                    if predict_tag != t_i:
                        error = self.ftpl_instantialize(sent_list[j], i, predict_tag)
                        # error_seql = self.sequentialize(error)
                        correct = self.ftpl_instantialize(sent_list[j], i, t_i)
                        # correct_seql = self.sequentialize(correct)
                        for f_error in error:
                            if f_error in self.epsilon:
                                error_id = self.epsilon.index(f_error)
                                self.W[error_id] -= 1

                        for f_correct in correct:
                            if f_correct in self.epsilon:
                                correct_id = self.epsilon.index(f_correct)
                                self.W[correct_id] += 1

            # evaluate
        # return self.W

    def score(self, f_vec):
        score = 0
        for f in f_vec:
            if f in self.epsilon:
                f_id = self.epsilon.index(f)
                score += self.W[f_id]
        return score



    def predict(self, sent, pos_index):
        max_score = -1e10
        best_tag = ''
        for t in self.tag_list:
            tmp_features = self.ftpl_instantialize(sent, pos_index, t)
            final_vec = self.sequentialize(tmp_features)

            score = self.score(final_vec)
            if score > (max_score + 1e-10):
                max_score = score
                best_tag = t
        return best_tag

