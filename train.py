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


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def word_fasttext(weights=None):
    word_input = Input(shape=(config.maxlen,), dtype="int32")
    embed_train_X = Embedding(config.word_num, config.embed_dim, input_length=config.maxlen)(word_input)
    output = Dense(config.num_classes, activation='softmax')(GlobalAveragePooling1D()(embed_train_X))
    model = Model(inputs=word_input, outputs=output)
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=config.lr),
                metrics=[precision, recall])
    if weights is not None:
        model.set_weights(weights)
    return model

def word_char_fasttext(weights=None):
    word_input = Input(shape=(config.maxlen,), dtype="int32")
    char_input = Input(shape=(config.char_maxlen,), dtype="int32")
    embed_word_X = Embedding(config.word_num, config.embed_dim, input_length=config.maxlen, name="word_embedding")(word_input)
    embed_char_X = Embedding(config.char_num, config.embed_dim, input_length=config.char_maxlen, name="char_embedding")(char_input)

    global_words = GlobalAveragePooling1D()(embed_word_X)
    global_chars = GlobalAveragePooling1D()(embed_char_X)
    feat = concatenate([global_words, global_chars],axis=1)
    output = Dense(config.num_classes, activation='softmax')(feat)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(lr=config.lr),
                metrics=[precision, recall])
    if weights is not None:
        model.set_weights(weights)
    return model

def make_batches( size, batch_size):

    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1) * batch_size)) for i in range(0, nb_batch)]


def get_batch_generator(X, Y, batch_size,Shuffle = True):
    index_array = np.arange(X.shape[0])
    if Shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(X.shape[0]-1,batch_size)

    while 1:
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = X[batch_ids]
            Y_batch = Y[batch_ids]
            yield (X_batch, Y_batch)

def get_batch_generator_word_char(word_X, char_X , Y, batch_size,Shuffle = True):
    index_array = np.arange(Y.shape[0])
    if Shuffle:
        np.random.shuffle(index_array)
    batches = make_batches(Y.shape[0]-1, batch_size)

    while 1:
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]

            X_batch = [word_X[batch_ids],char_X[batch_ids]]
            Y_batch = Y[batch_ids]
            yield  (X_batch, Y_batch)

def word_train_by_generator():

    model = word_fasttext()
    train_X,train_Y = pickle.load(open(config.word_train_pk,"rb"))
    test_X, test_Y = pickle.load(open(config.word_test_pk,"rb"))
    # print(train_X)
    # print(test_X)
    for i in range(config.epochs):
        print("epoch"+str(i)+":")
        if i == 8:
            K.set_value(model.optimizer.lr, 0.001)
        model.fit_generator(
            get_batch_generator(train_X, train_Y, config.batch_size),
            epochs = 1,
            steps_per_epoch=int(train_X.shape[0]/config.batch_size),
            validation_data=(test_X, test_Y)
            )
        c, p, r = model.evaluate(test_X, test_Y, batch_size=config.batch_size, verbose=config.verbose)
        model.save(config.model_dir+'/%s_epoch_%s_%s.bin.gz'%("fasttext", i, str(2*p*r/(p+r))[:6]))

def word_char_train_by_generator():
    model = word_char_fasttext()
    word_train_X, train_Y = pickle.load(open(config.word_train_pk,"rb"))
    print(word_train_X)
    word_test_X, test_Y = pickle.load(open(config.word_test_pk,"rb"))
    char_train_X = pickle.load(open(config.char_train_pk,"rb"))
    char_test_X = pickle.load(open(config.char_test_pk,"rb"))


    for i in range(config.epochs):
        print("epoch"+str(i)+":")
        if i == 8:
            K.set_value(model.optimizer.lr, 0.0001)
        model.fit_generator(
            get_batch_generator_word_char(word_train_X, char_train_X, train_Y, config.batch_size),
            epochs = 1,
            steps_per_epoch=int(train_Y.shape[0]/config.batch_size),
            validation_data=([word_test_X, char_test_X], test_Y)
            )
        c, p, r = model.evaluate([word_test_X, char_test_X], test_Y, batch_size=config.batch_size, verbose=config.verbose)
        model.save(config.model_dir+'/%s_epoch_%s_%s.bin.gz'%("fasttext", i, str(2*p*r/(p+r))[:6]))


if __name__ == "__main__":
    word_char_train_by_generator()
# word_train_by_generator()