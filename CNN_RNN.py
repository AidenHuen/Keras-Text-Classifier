# encoding:utf-8

from train import *
def convs_block(data, convs = [3,4,5], f = 256, name = "conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)

def get_model():
    word_input = Input(shape=(config.maxlen,), dtype="int32")
    char_input = Input(shape=(config.char_maxlen,), dtype="int32")
    embed_word_X = Embedding(config.word_num, config.embed_dim, input_length=config.maxlen, name="word_embedding")(word_input)
    embed_char_X = Embedding(config.char_num, config.embed_dim, input_length=config.char_maxlen, name="char_embedding")(char_input)
    trans_word = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embed_word_X))))
    trans_char = Activation(activation="relu")(BatchNormalization()((TimeDistributed(Dense(256))(embed_char_X))))

    # word rnn
    word_bigru = Bidirectional(GRU(128, return_sequences=True))(trans_word)
    avg_pool = GlobalAveragePooling1D()(word_bigru)
    max_pool = GlobalMaxPooling1D()(word_bigru)
    word_feat = concatenate([avg_pool, max_pool], axis=1)

    # char cnn
    char_feat = convs_block(trans_char, convs=[1, 2, 3, 4, 5], f=256, name="char_conv")

    conc = concatenate([word_feat, char_feat])
    dropfeat = Dropout(0.4)(conc)
    fc = Activation(activation='relu')(BatchNormalization()(Dense(128)(dropfeat)))
    output = Dense(4, activation="softmax")(fc)
    model = Model(inputs=[word_input, char_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=[precision, recall])
    return model



def train_cnn_rnn():
    model = get_model()
    word_train_X, train_Y = pickle.load(open(config.word_train_pk, "rb"))
    word_test_X, test_Y = pickle.load(open(config.word_test_pk, "rb"))
    char_train_X = pickle.load(open(config.char_train_pk, "rb"))
    char_test_X = pickle.load(open(config.char_test_pk, "rb"))
    print(word_train_X.shape,word_test_X.shape)
    print(char_train_X.shape,char_test_X.shape)
    print(test_Y.shape)
    for i in range(config.epochs):
        print("epoch" + str(i) + ":")
        if i == 8:
            K.set_value(model.optimizer.lr, 0.0001)
        model.fit_generator(
            get_batch_generator_word_char(word_train_X, char_train_X, train_Y, config.batch_size),
            epochs=1,
            steps_per_epoch=int(train_Y.shape[0] / config.batch_size),
            validation_data=([word_test_X, char_test_X], test_Y)
        )
        c, p, r = model.evaluate([word_test_X, char_test_X], test_Y, batch_size=config.batch_size,
                                 verbose=config.verbose)
        print("f1:", 2 * p * r / (p + r))
        model.save(config.model_dir + '/%s_epoch_%s_%s.h5' % ("cnn_rnn", i, str(2 * p * r / (p + r))[:6]))

train_cnn_rnn()