from collections import Counter

class Loader:
    def __init__(self, datapath):
        self.vocab_fre_dict, self.tag_fre_dict, self.sent_list, self.sent_tag_list, self.align_dict = self.load(datapath)
        # a = sorted(self.tag_fre_dict.items(), key=lambda x: x[1], reverse=True)
        # self.tag_list = ['<BOS>', '<EOS>'] + list({k:v for k,v in sorted(self.tag_fre_dict.items(), key=lambda x: x[1],reverse=True)})
        # self.vocab_list = ['<UNK>'] + list({k:v for k,v in sorted(self.vocab_fre_dict.items(), key=lambda x:x[1], reverse=True)})

        self.tag_list = list(
            {k: v for k, v in sorted(self.tag_fre_dict.items(), key=lambda x: x[1], reverse=True)})
        self.vocab_list = ['<UNK>'] + list(
            {k: v for k, v in sorted(self.vocab_fre_dict.items(), key=lambda x: x[1], reverse=True)})

        self.N = len(self.tag_list)
        self.V = len(self.vocab_list)

    def load(self, datapath):
        count = 0
        vocab_fre_dict = {}
        tag_fre_dict = {}
        sent_list = []
        sent_tag_list = []
        # sent_dict = {}
        tmp_sent_list = []
        tmp_tag_list = []

        # align_dict = {}

        with open(datapath, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line != '\n':
                    count += 1
                    # print(line)
                    line_list = line.split()
                    # print(line_list)
                    token = line_list[1]
                    tag = line_list[3]
                    if token not in vocab_fre_dict:
                        vocab_fre_dict[token] = 1
                    else:
                        vocab_fre_dict[token] += 1

                    if tag not in tag_fre_dict:
                        tag_fre_dict[tag] = 1
                    else:
                        tag_fre_dict[tag] += 1
                    tmp_tag_list.append(tag)
                    tmp_sent_list.append(token)
                else:
                    sent_list.append(tmp_sent_list)
                    sent_tag_list.append(tmp_tag_list)
                    tmp_sent_list = []
                    tmp_tag_list = []

                # if count >= 23:
                #     break

        tmp_align_list = []
        for i,sent in enumerate(sent_list):
            aaa = [pair for pair in zip(sent, sent_tag_list[i])]
            tmp_align_list.extend(aaa)
        align_dict = Counter(tmp_align_list)
        return vocab_fre_dict, tag_fre_dict, sent_list, sent_tag_list, align_dict

    def tag2id(self, tag):
        return self.tag_list.index(tag)

    def token2id(self, token):
        return self.vocab_list.index(token)

    def id2tag(self, tagid):
        return self.tag_list[tagid]

    def id2token(self, tokenid):
        return self.vocab_list[tokenid]

# if __name__ == '__main__':
#     train_path = '../train.conll'
#     dataset =Loader(train_path)
#     print(dataset.N)
#     print(dataset.V)