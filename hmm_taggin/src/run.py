from hmm_taggin.src.dataloader import Loader
import hmm_taggin.src.utils as utils
from hmm_taggin.src.myHMM import Hmm

if __name__ == '__main__':
    training_path = '../train.conll'
    dataset = Loader(training_path)

    print(len(dataset.sent_list))
    print()

    N = dataset.N
    V = dataset.V

    tag_fre_dict = dataset.tag_fre_dict
    vocab_fre_dict = dataset.vocab_fre_dict
    sent_list = dataset.sent_list
    sent_tag_list = dataset.sent_tag_list

    print(N)
    print(V)

    # utils.mle4ABPi(tag_fre_dict, vocab_fre_dict, sent_list, sent_tag_list, N, V)
    #
    print(dataset.tag_list)
    print(dataset.vocab_list)
    # print(dataset.sent_tag_list[:3])
    # print(dataset.sent_list)
    print(dataset.align_dict)


    hmm1 = Hmm(dataset=dataset, N=N, V=V, alpha=0)
    # print(hmm1.A[0])
    # print(len(hmm1.A[0]))
    # print(hmm1.A[:10])

    '''
        test B
    '''
    print(list(hmm1.B[2]))
    print(len(list(hmm1.B[2])))
    print([i for i in range(len(list(hmm1.B[2]))) if list(hmm1.B[2])[i] != 0])

    print(dataset.align_dict[(dataset.id2token(26), dataset.id2tag(2))])
    print((dataset.id2token(26), dataset.id2tag(2)))
    print(f'a.shape: {hmm1.A.shape}')
    print(f'b.shape: {hmm1.B.shape}')

    print(hmm1.B[:3])
    print('-------------------------')
    print(hmm1.A[:,0])
    print(len(hmm1.A[:,0]))








