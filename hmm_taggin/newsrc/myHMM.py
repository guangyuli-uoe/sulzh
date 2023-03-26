import numpy as np
from collections import Counter
'''

    components of a hmm (tagging)
        1. a set of states (tags for pos)
        2. an output alphabet (words)
        3. intial state (beginning of sentence)
        4. state transition probabilities ( p(t_i | t_i-1) )
        5. symbol emission probabilities ( p(w_i | t_i) )
'''

class Hmm:
    def __init__(self, N, V, dataset, alpha):
        '''
        :param N: num of hidden states
        :param V: vocab size
        :param T: timestep
        :param A: transition matrix
        :param B: emittion matrix
        :param Pi: initial matrix(vector)
        '''
        self.N = N
        self.V = V
        # self.T = T
        self.alpha = alpha
        self.A = np.zeros([self.N, self.N])
        self.B = np.zeros([self.N, self.V])
        self.Pi = np.zeros(self.N)
        self.dataset = dataset
        self.mle_train()

    def mle_train(self): # for training / get parameters

        tag_fre_dict = self.dataset.tag_fre_dict
        sent_tag_list = self.dataset.sent_tag_list

        # A-[N, N]: P(q_t+1 | q_t)
        for rid,row in enumerate(self.A):
            r_tag = self.dataset.id2tag(rid)
            count_r = tag_fre_dict[r_tag]
            for cid,column in enumerate(row):
                c_tag = self.dataset.id2tag(cid)
                count_rc = 0
                for sent_tag in sent_tag_list:
                    if r_tag in sent_tag:
                        tmp2_list = [pair for pair in zip(sent_tag[:-1], sent_tag[1:])]
                        tmp2_fre_dict = Counter(tmp2_list)
                        count_rc += tmp2_fre_dict[(r_tag, c_tag)]
                p_r2c = (count_rc + self.alpha) / (count_r + self.alpha * self.V)
                self.A[rid, cid] = p_r2c
        # B-[N, V]: P(o_t | q_t)
        for ridb, rowb in enumerate(self.B):
            tagb = self.dataset.id2tag(ridb)
            num_tag = tag_fre_dict[tagb]
            for cidb, columnb in enumerate(rowb):
                tokenb = self.dataset.id2token(cidb)
                num_tag_token = self.dataset.align_dict[(tokenb, tagb)]
                self.B[ridb, cidb] = (num_tag_token + self.alpha) / (num_tag + self.alpha * self.V)

        # Pi-[N]: P(q | <BOS>)
        for i in range(self.N):
            tag_i = self.dataset.id2tag(i)
            first_tag_list = [sent_tag[0] for sent_tag in sent_tag_list]
            tmp_1_tag_dict = Counter(first_tag_list)
            self.Pi[i] = (tmp_1_tag_dict[tag_i] + self.alpha) / (len(sent_tag_list) + self.V * self.alpha)


    def viterbi_predict(self, input_seq, T):
        tmp_matrix = np.zeros([self.N, T])
        backpointer = np.zeros([self.N, T])
        if input_seq[0] not in self.dataset.vocab_list:
            o1 = 0
        else:
            o1 = self.dataset.token2id(input_seq[0])
        # initialization step
        for i in range(len(self.Pi)):
            tmp_matrix[i, 0] = self.Pi[i] * self.B[i, o1]
            backpointer[i, 0] = 0
        # recursion step
        for t in range(1, len(input_seq)):
            if input_seq[t] not in self.dataset.vocab_list:
                o_t = 0
            else:
                o_t = self.dataset.token2id(input_seq[t])

            for s in range(self.N):
                # tmp_matrix[k, j] =
                # tmp_p = tmp_matrix[k, t-1] * self.A[k, t-1] * self.B[k, o_t]
                # max_p = 0
                candidates = []
                for j in range(self.N):
                    tmp_p = tmp_matrix[j, t-1] * self.A[j, s] * self.B[s, o_t]
                    candidates.append(tmp_p)
                    # if tmp_p > max_p:
                    #     max_p = tmp_p
                tmp_matrix[s, t] = np.max(candidates)
                backpointer[s, t] = np.argmax(candidates)

        best_path = []
        best_prob = []
        for column in range(tmp_matrix.shape[1]):
            max_p_column = np.max(tmp_matrix[:,column])
            ind = np.argmax(tmp_matrix[:, column])
            best_prob.append(max_p_column)
            best_path.append(self.dataset.id2tag(ind))

        return best_path, best_prob

    def evaluate(self, gold_tag_seq, input_tag_seq):
        correct_num = 0
        for i in range(len(gold_tag_seq)):
            if gold_tag_seq[i] == input_tag_seq[i]:
                correct_num += 1
        acc = correct_num / len(gold_tag_seq)
        return acc
        # for pair in zip(gold_tag_seq, input_tag_seq):

        # for i,gold in enumerate(gold_tag_seq):
        #     if gold == input_tag_seq[i]:
        #         correct_num += 1
        #
        # print(correct_num)
        # acc = correct_num / len(gold_tag_seq)
        # return acc





