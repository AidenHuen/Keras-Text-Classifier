# encoding: utf-8
import gzip, pickle
from FastTest import FastText
import config
# from keras.models import  load_model

def train_supervised(train_path,test_path, lr=0.01, dim=100, epoch=5, minCount=1, wordNgrams=2, label='__label__', verbose=1, maxlen=700, batch_size = 128):
    fastText = FastText()
    # fastText.train(train_path,test_path)
    fastText.train_by_generator(train_path,test_path)
    return fastText

def load_model(path):
    with gzip.open(path, 'rb') as f:
        args = pickle.load(f)

    fastText = FastText(args=args)
    return fastText



if __name__ == '__main__':

    text = '''birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta high school in
    lakewood township new jersey . it was founded by rabbi shmuel zalmen stein in 2001 after his
    father rabbi chaim stein asked him to open a branch of telshe yeshiva in lakewood .
    as of the 2009-10 school year the school had an enrollment of 76 students and 6 . 6 classroom
    teachers ( on a fte basis ) for a student–teacher ratio of 11 . 5 1 .'''

    model = train_supervised(config.train_path,config.test_path,maxlen=1400, wordNgrams=2,
                             lr=0.01, epoch=10, minCount=5, batch_size = 128)


    #
    # print('Predict:', model2.predict(text, k=3))
