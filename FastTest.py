# encoding:utf-8
import keras
from keras.utils import to_categorical
from keras.preprocessing import sequence

from text2dataset import Text2Dataset
from keras.models import *
from keras.layers import *
import gzip, pickle
import  keras.backend as K
import numpy as np
import config
import random

class FastText:
    def __init__(self,args=None):
        if args is None:
            self.text2Dataset = Text2Dataset()
            self.maxlen = config.maxlen
            self.batch_size = config.batch_size
            self.embedding_dims = config.embed_dim
            self.epochs = config.epoch
            self.lr = config.lr
            self.verbose = config.verbose
            self.num_classes = config.num_classes
            self.max_features = None
            self.model = None
        else:
            (wordNgrams, label_prefix, minCount, word2idx, label2idx, token_indice,
             self.max_features, self.maxlen, self.batch_size, self.embedding_dims,
             self.epochs, self.lr, self.num_classes, model_weights) = args

            self.text2Dataset = Text2Dataset()
            self.text2Dataset.words2idx = lambda words: [word2idx[word] for word in words.split() if word in word2idx]
            self.text2Dataset.label2idx = label2idx
            self.text2Dataset.idx2label = {label2idx[label]: label for label in label2idx}
            self.text2Dataset.token_indice = token_indice
            self.model = self.build_model(model_weights)

    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall



    def build_model(self, weights=None):
        word_input = Input(shape=(self.maxlen,), dtype="int32")
        embed_train_X = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(word_input)
        output = Dense(self.num_classes, activation='softmax')(GlobalAveragePooling1D()(embed_train_X))
        model = Model(inputs=word_input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=self.lr),
                metrics=[self.precision, self.recall])
        if weights is not None:
            model.set_weights(weights)

        return model

    def make_batches(self, size, batch_size):
        nb_batch = int(np.ceil(size/float(batch_size)))
        return [(i*batch_size, min(size, (i+1)* batch_size)) for i in range(0, nb_batch)]

    def get_batch_generator(self, X, Y, batch_size,Shuffle = True):
        index_array = np.arange(X.shape[0])
        if Shuffle:
            np.random.shuffle(index_array)


        batches = self.make_batches(X.shape[0]-1,batch_size)
        while 1:
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                X_batch = X[batch_ids]
                Y_batch = Y[batch_ids]
                yield (X_batch, Y_batch)



    def train(self, train_path,test_path):

        train_X, train_Y = self.text2Dataset.load_word_Train(train_path)
        train_X, train_Y = self.data_padding_categorical(train_X,train_Y)
        self.max_features = self.text2Dataset.word_num
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=self.lr),
                metrics=[self.precision, self.recall]
                           )
        test_X,test_Y = self.text2Dataset.load_word_Test(test_path)
        test_X,test_Y = self.data_padding_categorical(test_X,test_Y)
        self.model.fit(train_X, train_Y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=self.verbose,
                       validation_data= (test_X,test_Y)
   )
        return self

    def train_by_generator(self, train_path, test_path):
        train_X, train_Y = self.text2Dataset.load_word_Train(train_path)
        train_X, train_Y = self.data_padding_categorical(train_X,train_Y)
        self.max_features = self.text2Dataset.word_num
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=self.lr),
                metrics=[self.precision, self.recall]
                           )
        test_X,test_Y = self.text2Dataset.load_word_Test(test_path)
        test_X,test_Y = self.data_padding_categorical(test_X,test_Y)
        for i in range(self.epochs):
            print("epoch"+str(i)+":")
            if i == 8:
                K.set_value(self.model.optimizer.lr, 0.001)
            self.model.fit_generator(
                self.get_batch_generator(train_X,train_Y,self.batch_size),
                epochs = 1,
                steps_per_epoch=int(train_X.shape[0]/self.batch_size),
                validation_data=(test_X, test_Y)
            )
            c, p, r = self.model.evaluate(test_X, test_Y, batch_size=self.batch_size, verbose=self.verbose)
            self.save_model('tmp/%s_epoch_%s_%s.bin.gz'%("fasttext", i, str(2*p*r/(p+r))[:6]))
        return self

    def data_padding_categorical(self, X, Y):
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        Y = to_categorical(Y, self.num_classes)
        return X, Y

    def test(self, text_path, verbose=1):
        test_X, test_Y = self.text2Dataset.load_word_Test(text_path)
        test_X, test_Y  = self.data_padding_categorical(test_X, test_Y)
        c, p, r = self.model.evaluate(test_X, test_Y, batch_size=self.batch_size, verbose=verbose)

        print("N\t" + str(len(test_Y)))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))
        return str(len(test_Y)), p, r

    def save_model(self, path):
        args = (self.text2Dataset.wordNgrams, self.text2Dataset.label_prefix,
                self.text2Dataset.minCount, self.text2Dataset.word2idx,
                self.text2Dataset.label2idx, self.text2Dataset.token_indice,
                self.max_features, self.maxlen, self.batch_size, self.embedding_dims,
                self.epochs, self.lr, self.num_classes, self.model.get_weights())
        with gzip.open(path, 'wb') as f:
            pickle.dump(args, f)
        print("model saved !")

    def predict(self, text, k=1):
        text = ','.join([words for words in text.split(',')]).strip().replace('\n', '')
        X = self.text2Dataset.words2idx(text)
        X = self.text2Dataset.add_ngram([X])
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        predict = self.model.predict(X).flatten()
        results = [(self.text2Dataset.idx2label[idx], predict[idx]) for idx in range(len(predict))]
        return sorted(results, key=lambda item: item[1], reverse=True)[:k]