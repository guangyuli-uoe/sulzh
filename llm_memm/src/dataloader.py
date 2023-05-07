from collections import Counter

class DataLoader():
    def __init__(self, datapath, batch_size):
        self.datapath = datapath
        self.batch_size = batch_size
        self.sent_word_list, self.sent_tag_list, self.sent_char_list = self.preprocess()
        # self.ns = len(self.word_list)
        # self.nw = len([w for sent in self.word_list for w in sent])
        self.word_fre_dict, self.tag_fre_dict, self.char_fre_dict = self.build_dict()
        self.word_dict, self.tag_dict, self.char_dict = self.buil_vocab()

        # # 词汇字典
        # self.wdict = {w: i for i, w in enumerate(self.words)}
        # # 词性字典
        # self.tdict = {t: i for i, t in enumerate(self.tags)}
        # # 字符字典
        # self.cdict = {c: i for i, c in enumerate(self.chars)}
        self.seqlized_sent_word_list = self.sequentialize(level=0)
        self.seqlized_sent_tag_list = self.sequentialize(level=1)
        self.seqlized_sent_char_list = self.sequentialize(level=2)

        self.batch_data = self.build_batch()

    def preprocess(self):
        tmp_tag_list = []
        tmp_sent_list = []
        tmp_char_list = []

        word_list = []
        tag_list = []
        char_list = []
        with open(self.datapath, 'r', encoding='utf-8') as fr:
            for line in fr:
                if line != '\n':
                    line_list = line.split()
                    # print(line_list)
                    token = line_list[1]
                    tag = line_list[3]
                    char = [c for c in token]
                    tmp_tag_list.append(tag)
                    tmp_sent_list.append(token)
                    tmp_char_list.append(char)
                    # print(tmp_tag_list)
                else:
                    word_list.append(tmp_sent_list)
                    tag_list.append(tmp_tag_list)
                    char_list.append(tmp_char_list)
                    tmp_sent_list = []
                    tmp_tag_list = []
                    tmp_char_list = []
        return word_list, tag_list, char_list

    def __repr__(self):
        info = "%s(\n" % self.__class__.__name__
        info += "  num of sentences: %d\n" % len(self.sent_word_list)
        info += "  num of uniq words: %d\n" % len(self.word_dict)
        info += "  num of uniq tags: %d\n" % len(self.tag_dict)
        info += "  num of uniq chars: %d\n" % len(self.char_dict)
        info += ")"
        return info

    def build_dict(self):
        # words = sorted(set(w for wordseq in self.sent_word_list for w in wordseq))
        # tags = sorted(set(t for tagseq in self.sent_tag_list for t in tagseq))
        # chars = sorted(set(''.join(words)))
        # set() = = {''.join(words)}
        # print(words)
        # print(tags)
        # print(chars)

        all_words = sorted([w for sent in self.sent_word_list for w in sent])
        all_tags = sorted([t for sent in self.sent_tag_list for t in sent])
        all_chars = sorted([c for c in ''.join(all_words)])

        word_fre_dict = Counter(all_words)
        # print(word_dict)
        # print(len(all_words))
        # print(len(word_dict))
        tag_fre_dict = Counter(all_tags)
        char_fre_dict = Counter(all_chars)
        return word_fre_dict, tag_fre_dict, char_fre_dict

    def buil_vocab(self):
        word_dict = {k:i for i,k in enumerate(self.word_fre_dict)}
        # print(word_dict)
        tag_dict = {k:i for i,k in enumerate(self.tag_fre_dict)}
        char_dict = {k:i for i,k in enumerate(self.char_fre_dict)}
        return word_dict, tag_dict, char_dict

    # def seq_op(self, org_list, vocab_dict):
    #     seqlized_list = []
    #     for sent in org_list:
    #         seqlized_list.append([vocab_dict[i] for i in sent])
    #     return seqlized_list

    def sequentialize(self, level):
        if level == 0:
            vocab_dict = self.word_dict
            org_list = self.sent_word_list
            seqlized_list = []
            for sent in org_list:
                # print(f'sent: {sent}')
                seqlized_list.append([vocab_dict[i] for i in sent])
        elif level == 1:
            vocab_dict = self.tag_dict
            org_list = self.sent_tag_list
            seqlized_list = []
            for sent in org_list:
                seqlized_list.append([vocab_dict[i] for i in sent])
        elif level == 2:
            vocab_dict = self.char_dict
            org_list = self.sent_char_list
            seqlized_list = []
            for sent in org_list:
                # print(f'sent: {sent}')
                for word in sent:
                    seqlized_list.append([vocab_dict[i] for i in word])

        else:
            raise ValueError

        return seqlized_list

    def build_batch(self):
        num_batch = len(self.sent_word_list) // self.batch_size
        all_data = zip(self.sent_word_list, self.sent_tag_list)
        all_data_list = [i for i in all_data]
        batch_data = []
        for i in range(num_batch):
            if i == 0:
                begin = 0
            else:
                begin = i + self.batch_size
            batch = all_data_list[begin:self.batch_size]
            batch_data.append(batch)
        return batch_data
