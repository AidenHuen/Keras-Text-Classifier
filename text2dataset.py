# encoding:utf-8

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing import sequence
import pickle,gzip,codecs
import config
class Text2Dataset:
    def __init__(self):
        self.wordNgrams = 2
        self.label_prefix = config.label
        self.minCount = config.minCount

        self.word2idx = None

        self.label2idx = {"自动摘要":0, "机器翻译":1, "机器作者":2, "人类作者":3}
        self.idx2label = None
        self.train_X = None
        self.train_y = None
        self.word_num = None
        self.char_num = None
        self.token_indice = None
        self.char2idx = None
        self.words2idx = None

    def create_ngram_set(self, input_list, ngram_value=2):
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def create_ngram_dict(self, input_list, ngram_value=2):
        tuple_list = list(zip(*[input_list[i:] for i in range(ngram_value)]))
        sample_tuple_dict = {}
        for tp in tuple_list:
            try:
                sample_tuple_dict[tp] += 1
            except:
                sample_tuple_dict[tp] = 1
        print(sample_tuple_dict)
        return sample_tuple_dict

    def add_ngram(self, sequences):
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, self.wordNgrams + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in self.token_indice:
                        new_list.append(self.token_indice[ngram])
            new_sequences.append(new_list)
        return new_sequences

    def text2List(self, text_path):
        with open(text_path) as f:
            lines = f.readlines()
        getLabel = lambda line: [words.strip() for words in line.split(',') if self.label_prefix in words][0]
        getWords = lambda line: (','.join([words for words in line.split(',') if self.label_prefix not in words])
                                 .strip().replace('\n', ''))

        label_list = [getLabel(line) for line in lines]
        words_list = [getWords(line) for line in lines]

        return words_list, label_list

    def take_data_label_csv(self, text_path):
        """
        pandas读取csv格式数据
        :param text_path:
        :return:
        """
        data = pd.read_csv(text_path, sep='\t')
        return data.content.values[:5000], data.label.values[:5000]


    def load_char_Train(self, text_path):
        words_list, label_list = self.take_data_label_csv(text_path)
        chars_list = [words.replace(" ", "") for words in words_list]
        del words_list
        # chars_list = [chars.decode("utf-8") for chars in chars_list]
        chars_list = [chars for chars in chars_list]
        chars_all = [char for char in ''.join(chars_list)]
        self.char2idx = {char: idx for idx, char in enumerate(set(chars_all))}
        self.idx2label = {self.label2idx[label]: label for label in self.label2idx}

        del chars_all

        char_X = []
        for chars in chars_list:
            idx = [self.char2idx[char] for char in chars if char in self.char2idx]
            char_X.append(idx)

        self.char_num = len(self.char2idx)
        if self.wordNgrams > 1:
            print('Adding {}-gram features'.format(self.wordNgrams))
            ngram_set = set()
            for input_list in char_X:
                for i in range(2, self.wordNgrams + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            start_index = self.char_num + 1
            self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {self.token_indice[k]: k for k in self.token_indice}
            self.char_num = np.max(list(indice_token.keys())) + 1
            char_X = self.add_ngram(char_X)

        return char_X

    def load_word_Train(self, text_path):

        words_list, label_list = self.take_data_label_csv(text_path)
        # self.idx2label = {self.label2idx[label]: label for label in self.label2idx}
        self.word2idx = pickle.load(open(config.word_vocab),t)
        self.words2idx = lambda words: [self.word2idx[word] for word in words.split() if word in self.word2idx]
        # print(self.word2idx)
        self.train_X = [self.words2idx(words) for words in words_list]
        self.train_y = [self.label2idx[label] for label in label_list]
        self.word_num = len(self.word2idx)
        if self.wordNgrams > 1:
            print('Adding {}-gram features'.format(self.wordNgrams))
            ngram_set = set()
            for input_list in self.train_X:
                for i in range(2, self.wordNgrams + 1):
                    dict_ngram_ = self.create_ngram_dict(input_list, ngram_value=i)
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            start_index = self.word_num + 1
            self.token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {self.token_indice[k]: k for k in self.token_indice}

            self.word_num = np.max(list(indice_token.keys())) + 1

            self.train_X = self.add_ngram(self.train_X)

        return self.train_X, self.train_y


    def load_word_Test(self, text_path):
        words_list, label_list = self.take_data_label_csv(text_path)
        test_X = [self.words2idx(words) for words in words_list]
        test_y = [self.label2idx[label] for label in label_list]
        test_X = self.add_ngram(test_X)
        return test_X, test_y

    def load_char_Test(self,text_path):
        words_list, label_list = self.take_data_label_csv(text_path)
        chars_list = [words.replace(" ", "") for words in words_list]
        del words_list
        # chars_list = [chars.decode("utf-8") for chars in chars_list]
        chars_list = [chars for chars in chars_list]
        char_X = []
        for chars in chars_list:
            idx = [self.char2idx[char] for char in chars if char in self.char2idx]
            char_X.append(idx)
        char_X = self.add_ngram(char_X)
        return char_X

    def save_char__pk(self):

        char_train = self.load_char_Train(config.train_path)
        char_train = sequence.pad_sequences(char_train, maxlen=config.char_maxlen)

        char_test = self.load_char_Test(config.test_path)
        char_test = sequence.pad_sequences(char_test, maxlen=config.char_maxlen)
        pickle.dump(char_train, open(config.char_train_pk, "wb"))
        pickle.dump(char_test, open(config.char_test_pk, "wb"))
        print(self.char_num)
        print(".pk saved finish!!")

    def save_word_pk(self):
        train_X,train_Y = self.load_word_Train(config.train_path)
        train_X = sequence.pad_sequences(train_X, maxlen=config.maxlen)
        train_Y = to_categorical(train_Y)

        test_X, test_Y = self.load_word_Test(config.test_path)
        test_X = sequence.pad_sequences(test_X, maxlen=config.maxlen)
        test_Y = to_categorical(test_Y)
        pickle.dump((train_X,train_Y), open(config.word_train_pk, "wb"))
        pickle.dump((test_X,test_Y), open(config.word_test_pk, "wb"))
        print(train_X.shape,train_Y.shape)
        print(test_X.shape,test_Y.shape)
        print(self.word_num)
        print(".pk saved finish!!")

    # def save_char_train_test_pk(self):
    def save_word2idf_pk(self):
        words_list, label_list = self.take_data_label_csv(config.train_path)
        self.idx2label = {self.label2idx[label]: label for label in self.label2idx}
        self.word2idx = {word: idx for idx, word in enumerate(set(' '.join(words_list).split()))}
        self.words2idx = lambda words: [self.word2idx[word] for word in words.split() if word in self.word2idx]
        pickle.dump(self.word2idx, open(config.word_vocab,"wb"))

if __name__ == "__main__":
    # Text2Dataset().save_word_train_test_pk()
    # Text2Dataset().save_char_train_test_pk()
    # Text2Dataset().save_word2idf_pk()
    Text2Dataset().load_word_Train(config.train_path)