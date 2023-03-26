import numpy as np


def mle4ABPi(tag_fre_dict, vocab_fre_dict, sent_list, sent_tag_list, N, V):
    print(len(tag_fre_dict))
    print(len(vocab_fre_dict))
    print(len(sent_list))

    total_tag = sum(tag_fre_dict.values())
    total_token = sum(vocab_fre_dict.values())
    print(total_token)
    print(total_tag)

    print(np.zeros([3, 3]))
    print(np.zeros(3))

    print(tag_fre_dict)



def viterbi(input_sentece, T, N, A, B): # for predicting: given O (token seq) predict Q (tag seq)
    tmp_matrix = np.zeros([N, T])
    backpointer = np.zeros([N, T])
    Pi = np.zeros(N)
    Pi[0] = 1
    print(Pi)

    for rid in range(N):
        tmp_matrix[rid, 0] = Pi[rid] * B[0, rid]


if __name__ == '__main__':
    # viterbi('a', 3, 4, 1, 1)

    n = 3
    for i in range(n):
        print(1)