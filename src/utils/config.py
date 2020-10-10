import os
import json
import torch

current_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(current_path)[0])[0]

data_path = root_path + '/data/THUCNews/'

train_path = root_path + '/data/train.csv'
test_path = root_path + '/data/test.csv'
valid_path = root_path + '/data/valid.csv'

label_path = root_path + '/data/label2id.json'
# bert_path = root_path + '/model/wobert_torch/'
# bert_path = root_path + '/model/goole_bert/'
# bert_path = root_path + '/model/roberta/'
bert_path = root_path + '/model/wonezha_L-12_H-768_A-12/'
# bert_path = root_path + 'model/bert-chinese-wwm-ext'

is_cuda = True
device = torch.device('cuda') if is_cuda else torch.device('cpu')

with open(root_path + '/data/stopwords.txt', "r", encoding='utf-8') as f:
    stopWords = [word.strip() for word in f.readlines()]

with open(label_path, 'r', encoding='utf-8') as f:
    label2id = json.load(f)

label_list = label2id.keys()
model_name = 'Bert'
save_path = root_path + '/model/saved_dict/bert_wo_cls.pt'
# bert
eps = 1e-8
learning_rate = 2e-5  # 学习率
embedding_pretrained = None
batch_size = 64
hidden_size = 768
num_epochs = 100
dropout = 0.3  # 随机失活
require_improvement = 1000 # 若超过1000batch效果还没提升，则提前结束训练
num_classes = len(label2id)  # 类别数
n_vocab = 50000  # 词表大小，在运行时赋值
embed = 300

# cnn
# learning_rate = 1e-3  # 学习率
# filter_sizes = (2, 3, 4)  # 卷积核尺寸
# num_filters = 256

# lstm
# hidden_size = 256
# num_layers = 2
# epochs = 64
# pad_size = 128

fast_path = root_path + '/model/fast.bin'
w2v_path = root_path + '/model/w2v.bin'

dict_path = root_path + '/data/vocab.bin'
log_path = root_path + '/logs/' + model_name

# rnn_att
hidden_size2 = 256