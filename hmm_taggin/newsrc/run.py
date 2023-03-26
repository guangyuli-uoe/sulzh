from hmm_taggin.newsrc import dataloader
from hmm_taggin.newsrc import myHMM

if __name__ == '__main__':
    training_path = '../train.conll'
    dataset = dataloader.Loader(training_path)
    N = dataset.N
    V = dataset.V
    hmm = myHMM.Hmm(N, V, dataset, alpha=0)
    # print(hmm.A[:3])

    # print([i for i in range(len(list(hmm.B[2]))) if list(hmm.B[2])[i] != 0])

    print(f'A.shape: {hmm.A.shape}')
    print(f'B.shape: {hmm.B.shape}')
    print(hmm.Pi)
    print(len(dataset.sent_tag_list))

    # A = hmm.A
    # B = hmm.B
    # Pi = hmm.Pi

    '''
        dev: sent_list[ [], [] ]
    '''
    # print(dataset.vocab_list)

    # dev_path = '../dev.conll'
    dev_path = '../train.conll'
    dev = dataloader.Loader(dev_path)

    input_list = dev.sent_list
    tag_list = dev.sent_tag_list
    # print(input_list[:10])

    count = 0
    total_acc = 0
    total_tag = 0
    total_correct_tag = 0
    for pair in zip(input_list, tag_list):
        count += 1
        # print(pair)
        input_seq = pair[0]
        gold_tag = pair[1]
        total_tag += len(gold_tag)
        T = len(input_seq)
        predict_tag, predict_prob = hmm.viterbi_predict(input_seq, T)
        print(input_seq)
        print(f'gold: {gold_tag}')
        print(f'predict: {predict_tag}')
        # acc = hmm.evaluate(gold_tag, input_seq)
        # print(acc)

        correct = 0
        for i in range(len(gold_tag)):
            if gold_tag[i] == predict_tag[i]:
                correct += 1
        acc = correct/len(gold_tag)
        print(f'acc: {acc}')
        total_acc += acc
        total_correct_tag += correct

        # if count >= 10:
        #     break

    print(f'acc/sent over dev: {total_acc/ len(input_list)}')
    print(f'acc/tag over dev: {total_correct_tag / total_tag}')

    '''
    
        alpha = 0
        acc/sent over dev: 0.4343904967934592
        acc/tag over dev: 0.37099306425008444
        
        alpha = 0.3
        acc/sent over dev: 0.7327572893724613
        acc/tag over dev: 0.7378326278344164
    '''