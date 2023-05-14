import numpy as np

a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(a)
print(a[0])

for i in range(1, 4):
    print(i)


    def predict(self, sentence, averaged_perceptron):
        N = len(sentence)
        useful_score = np.zeros((N, len(self.tag_list)))
        path = np.zeros((N, len(self.tag_list)), dtype="int")
        path[0] = -1
        # 句子的第一个词标注为每一个词性所构成的特征向量的集合
        first_feature_template_lst = [self.create_feature_template(sentence, 0, self.FIR, tag) for tag in
                                      self.tag_list]
        # 句子的第一个词标注为每一个词性对应的得分
        useful_score[0] = np.array([self.calculate_score(template, averaged_perceptron) for template in
                                    first_feature_template_lst])
        # 所有可能的当前词性和所有可能的前一个词性对应的得分构成的矩阵，
        # 有len(self.tag_lst)行，len(self.tag_lst)列，
        # 代表当前词和前一个词的特征向量的得分，这是一个二维的数组（矩阵）
        bigram_score_all_lst = np.array(
            [[self.calculate_score(template, averaged_perceptron) for template in templates] for templates in
             self.bigram_feature_template_all_lst])
        for i in range(1, N):
            uigram_template_feature = [self.create_uigram_feature_template(sentence, i, tag) for tag in self.tag_list]
            uigram_score = np.array(
                [self.calculate_score(template, averaged_perceptron) for template in uigram_template_feature])
            temp_score = (useful_score[i - 1] + bigram_score_all_lst).T + uigram_score
            path[i] = np.argmax(temp_score, axis=0)
            useful_score[i] = np.max(temp_score, axis=0)

        last = int(np.argmax(useful_score[-1]))
        last_tag = self.tag_list[last]
        predict_tag_lst = [last_tag]
        T = len(sentence) - 1
        for i in range(T, 0, -1):
            last = path[i][last]
            predict_tag_lst.insert(0, self.tag_list[last])
        return predict_tag_lst