import numpy as np

def viterbi(input_sentece, T, N, A, B): # for predicting: given O (token seq) predict Q (tag seq)
    tmp_matrix = np.zeros([N, T])
    backpointer = np.zeros([N, T])
    Pi = np.zeros(N)
    Pi[0] = 1
    print(Pi)

    for rid in range(N):
        tmp_matrix[rid, 0] = Pi[rid] * B[0, rid]