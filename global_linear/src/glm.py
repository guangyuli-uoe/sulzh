import numpy as np

class GlobalLinearModel():
    def __init__(self,dataset, devdataset):
        self.dataset = dataset
        self.dev = devdataset
        self.epsilon = self.create_feature_space()

    def ftpl_instantialize(self, sent, i, tags):
        features = []

        t_i = tags[i]
        t_i_pre = tags[i-1] if i != 0 else '#'

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

    def viterbi_predict(self, sentence):
        T = len(sentence)


