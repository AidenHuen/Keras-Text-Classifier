# encoding:utf-8
pk_dir = "/media/iiip/数据/Aiden/smp_data"
train_path = pk_dir+"/train.csv"
test_path = pk_dir+"/test.csv"


word_train_pk = pk_dir + "/pickle/train.pk"
word_test_pk = pk_dir + "/pickle/test.pk"
char_train_pk = pk_dir + "/pickle/char_train.pk"
char_test_pk = pk_dir + "/pickle/char_test.pk"

word_vocab = pk_dir + "/pickle/word_vocab.pk"

num_classes = 4
lr = 0.001
embed_dim = 200
epochs = 15
minCount = 1
wordNgrams = 1
label = '__label__'
verbose = 1
maxlen = 700
char_maxlen = 900
batch_size = 128
word_num = 384748 # 词数
char_num = 7412 # 字数


model_dir = '/media/iiip/文档/Aiden/model'
