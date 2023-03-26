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
        self.mle4AB()

    def mle4AB(self): # for training / get parameters
        '''
                row, []
                column, num/p
        :return:
        '''
        tag_fre_dict = self.dataset.tag_fre_dict
        # tag_fre_dict = dataset.tag_fre_dict
        # vocab_fre_dict = self.dataset.vocab_fre_dict
        # sent_list = self.dataset.sent_list
        sent_tag_list = self.dataset.sent_tag_list
        '''
            creating transition matrix A
        '''
        for rid,row in enumerate(self.A):
            r_tag = self.dataset.id2tag(rid)
            if rid == 0:
                count_r = 803
            elif rid == 1:
                count_r = 803
            else:
                count_r = tag_fre_dict[r_tag]
            for cid,column in enumerate(row):
                c_tag = self.dataset.id2tag(cid)

                count_rc = 0
                if rid == 0: # <bos> -> ?
                    if cid == 0:
                        count_rc = 0
                    elif cid == 1:
                        count_rc = 0
                    else:
                        first_tag_list = [sent_tag[0] for sent_tag in sent_tag_list]
                        tmp_1_tag_dict = Counter(first_tag_list)
                        if c_tag not in tmp_1_tag_dict:
                            count_rc = 0
                        else:
                            count_rc = tmp_1_tag_dict[c_tag]
                elif rid == 1: # <eos> -> ?
                    count_rc = 0
                else:
                    if cid == 0:
                        count_rc = 0
                    elif cid == 1:
                        last_tag_list = [sent_tag[-1] for sent_tag in sent_tag_list]
                        tmp_last_tag_dict = Counter(last_tag_list)
                        if c_tag not in tmp_last_tag_dict:
                            count_rc = 0
                        else:
                            count_rc = tmp_last_tag_dict[c_tag]
                    else:
                        for sent_tag in sent_tag_list:
                            if r_tag in sent_tag:
                                tmp2_list = [pair for pair in zip(sent_tag[:-1], sent_tag[1:])]
                                tmp2_fre_dict = Counter(tmp2_list)
                                count_rc += tmp2_fre_dict[(r_tag, c_tag)]

                if rid == 1:
                    p_r2c = count_rc / count_r
                elif rid == 0:
                    if cid == 0:
                        p_r2c = count_rc / count_r
                    elif cid == 1:
                        p_r2c = count_rc / count_r
                    else:
                        p_r2c = (count_rc + self.alpha) / (count_r + self.alpha * self.V)
                else:
                    p_r2c = (count_rc + self.alpha) / (count_r + self.alpha * self.V)
                if cid == 0:
                    p_r2c = count_rc / count_r
                self.A[rid, cid] = p_r2c

        '''
            creating emitting matrix B
        '''
        # alignments = [pair for pair in zip(sent_tag_list, sent_list)]
        for ridb, rowb in enumerate(self.B):
            tagb = self.dataset.id2tag(ridb)
            for cidb, columnb in enumerate(rowb):
                tokenb = self.dataset.id2token(cidb)
                if ridb == 0:
                    self.B[ridb, cidb] = 0
                elif ridb == 1:
                    self.B[ridb, cidb] = 0
                else:
                    num_tag = tag_fre_dict[tagb]

                    num_tag_token = 0
                    # for align in alignments:
                    #     if tagb in align[0]:
                    #         if tokenb in align[1]:
                    #             if align[0].index(tagb) == align[1].index(tokenb):
                    #                 num_tag_token += 1

                    # for align_id, tags in enumerate(sent_tag_list):
                    #     if tagb in tags:
                    #         if tokenb in sent_list[align_id]:
                    #             if tags.index(tagb) == sent_list[align_id].index(tokenb):
                    #                 num_tag_token += 1

                    num_tag_token = self.dataset.align_dict[(tokenb, tagb)]
                    # print(num_tag_token)

                    self.B[ridb, cidb] = (num_tag_token + self.alpha) / (num_tag + self.alpha * self.V)
                # self.B[ridb, cidb] = p_r2c
            # print('--------B---------------')

        '''
            creating initialization matrix Pi
        '''

