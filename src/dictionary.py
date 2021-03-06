import jieba
import os
import re
jieba.initialize()
from string import punctuation
from collections import Counter
from tqdm import tqdm
import joblib
import pandas as pd
from src.tools import processes_data, multiprocess_data


def clean_symbols(text):
    """
    对特殊符号做一些处理
    """
    text = re.sub('[0-9]+', " NUM ", str(text))
    text = re.sub('[!！]+', " ", text)
    #     text = re.sub('!', '', text)
    text = re.sub('[?？]+', " ", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’'!'[\\]^_`{|}~]+", " OOV ", text)
    return re.sub("\s+", " ", text)

class Dictionary:
    """构建字典"""

    def __init__(self, max_size=50000, start_end_token=True, min_count=2):
        self.max_size = max_size
        self.start_end_tokens = start_end_token
        self.pad_token = '<PAD>'
        self.min_count = min_count

    def build_dictionary(self, data):
        self.word2id, self.id2word, self.vocab_words, self.word2count = self._build_dictionary(data)
        self.vocabulary_size = len(self.vocab_words)

    def _build_dictionary(self, data):
        vocab_words = [self.pad_token, '<UNK>']
        vocab_size = 2
        if self.start_end_tokens:
            vocab_words += ['<SOS>', '<EOS>']
            vocab_size += 2
        data = [word for sen in tqdm(data) for word in sen.split(' ')]
        word2count = Counter(data)
        # 词典的容量
        if self.max_size:
            word2count = {word: count for word, count in word2count.most_common(self.max_size - vocab_size)}
        # 过滤低频词
        if self.min_count:
            word2count = {word: count for word, count in word2count.items() if count > self.min_count}
        vocab_words += list(sorted(word2count.keys()))
        word2id = dict(zip(vocab_words, range(len(vocab_words))))
        id2word = vocab_words

        return word2id, id2word, vocab_words, word2count

    def indexer(self, word):
        """根据词查询id"""
        try:
            id = self.word2id[word]
        except:
            id = self.word2id['<UNK>']
        return id


if __name__ == '__main__':
    data = pd.read_csv(config.train_path, sep='\t').dropna()
    test = pd.read_csv(config.test_path, sep='\t')
    valid = pd.read_csv(config.valid_path, sep='\t')
    data = pd.concat([train, test, valid], axis=0).dropna()
    word = True
    data = multiprocess_data(data,processes_data,worker=10,word=True)
    if word:
        data = data["cut_sentence"].values.tolist()
    else:
        data = data['raw_words'].values.tolist()
    dictionary = Dictionary()
    dictionary.build_dictionary(data)
    print(dictionary.vocab_words[:20])
    joblib.dump(dictionary, open('dict.pkl', 'wb'))
    # dictionary = joblib.load('dict.pkl')
    # id = dictionary.indexer('世界')
    # print(id)
