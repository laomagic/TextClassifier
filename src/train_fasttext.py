#!/usr/bin/env python
# coding:utf-8
import pandas as pd
from tqdm import tqdm
from fasttext import train_supervised
import fasttext
import os
from __init__ import *
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import create_logger, clean_symbols, query_cut, rm_stop_word

logger = create_logger(root_path + '/logs/Fasttext.log')
tqdm.pandas()


class Fasttext(object):
    """
    使用fasttext 训练文本分类的模型
    """

    def __init__(self,
                 train_raw_path=config.train_path,
                 test_raw_path=config.test_path,
                 valid_raw_path=config.valid_path,
                 model_train_file=root_path + '/data/fast_train.txt',
                 model_test_file=root_path + '/data/fast_test.txt',
                 model_valid_file=root_path + '/data/fast_valid.txt',
                 model_path=None):
        """
        初始化参数
        :param train_raw_path: 原始训练文件路径
        :param test_raw_path: 原始测试文件路径
        :param valid_raw_path: 原始验证文件路径
        :param model_train_file: 训练集文件路径
        :param model_test_file: 测试集文件路径
        :param model_valid_file: 验证集文件路径
        :param model_path: 模型路径
        """
        if model_path is None or not os.path.exists(model_path):
            self.Train_raw_data = pd.read_csv(train_raw_path, sep='\t').dropna()
            self.Test_raw_data = pd.read_csv(test_raw_path, sep='\t').dropna()
            self.Valid_raw_data = pd.read_csv(valid_raw_path, sep='\t').dropna()
            if not os.path.exists(model_train_file):
                self.data_process(self.Train_raw_data, model_train_file)
            if not os.path.exists(model_test_file):
                self.data_process(self.Test_raw_data, model_test_file)
            if not os.path.exists(model_valid_file):
                self.data_process(self.Valid_raw_data, model_valid_file)
            self.train(model_train_file, model_test_file, model_valid_file)  # 训练模型
        else:
            self.classifier = fasttext.load_model(model_path)  # 导入模型

    def data_process(self, raw_data, model_data_file):
        """
        数据的预处理
        :param raw_data: 原始文本dataframe
        :param model_data_file: 保存的数据文件路径
        :return: None
        """
        logger.info('Geneating data.')
        raw_data["sentence"] = raw_data['title'] + raw_data['content']
        raw_data['clean_sentence'] = raw_data['sentence'].progress_apply(clean_symbols)
        # 去除标点符号
        raw_data["cut_sentence"] = raw_data['clean_sentence'].progress_apply(query_cut)
        # 去除停用词
        raw_data['stop_sentence'] = raw_data["cut_sentence"].progress_apply(rm_stop_word)
        raw_data['stop_sentence'] = raw_data['stop_sentence'].progress_apply(lambda x: ' '.join(x))
        with open(model_data_file, 'w', encoding="utf-8") as f:
            for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
                outline = row['stop_sentence'] + "\t__label__" + row['label'] + "\n"  # 生成的训练数据
                f.write(outline)

    def train(self, model_train_file, model_test_file, model_valid_file):
        """
        训练集、验证集和测试集的文件路径
        :param model_train_file:
        :param model_test_file:
        :param model_valid_file:
        :return: None
        """
        # self.classifier = train_supervised(model_train_file,
        #                                    label="__label__",
        #                                    dim=300,
        #                                    epoch=50,
        #                                    lr=0.1,
        #                                    wordNgrams=2,
        #                                    loss='softmax',
        #                                    minCount=5,
        #                                    verbose=True)
        # 使用验证集自动调参的方式，没有验证集可以手动设置参数
        logger.info('Training model.')
        self.classifier = train_supervised(model_train_file, verbose=2,
                                           autotuneValidationFile=model_valid_file,
                                           autotuneModelSize="2M",
                                           autotuneDuration=600)
        logger.info('Saving model.')
        self.classifier.save_model(config.root_path + '/model/news.ftz')  # 保存模型
        # 测试模型
        self.test(model_train_file, model_test_file)

    def test(self, model_train_file, model_test_file):
        """
        训练模型的测试
        :param model_train_file: 训练集路径
        :param model_test_file: 测试集路径
        :return:
        """
        logger.info('Testing.')

        def score(result):
            """precision recall f1 score"""
            f1 = (result[1] * result[2] * 2) / (result[2] + result[1])
            precision = result[1]
            recall = result[2]
            return precision, recall, f1

        test_result = self.classifier.test(model_test_file)  # 测试集结果
        train_result = self.classifier.test(model_train_file)  # 训练集结果

        # 返回精确率和召回率、F1-score
        train_score = score(train_result)
        test_score = score(test_result)
        print('训练集的precision:{:.4f}  recall:{:.4f}  f1-score:{:.4f}'.format(*train_score))
        print('测试集的precision:{:.4f}  recall:{:.4f}  f1-score:{:.4f}'.format(*test_score))

    def predict(self, text):
        """
        处理输入的文本数据
        :return: label,score
        """
        logger.info('Predicting.')
        clean_text = clean_symbols(text)
        cut_text = query_cut(clean_text)  # 分词后的句子
        label, score = self.classifier.predict(' '.join(cut_text))
        res = {'label': label[0].split('__label__')[1],
               'score': score[0]}
        return res


if __name__ == '__main__':
    ft = Fasttext(model_path=config.root_path + '/model/news.ftz')
    res = ft.predict('9月3日，美国媒体报道，英国演员罗伯特·帕丁森的新冠病毒检测结果呈阳性，导致电影《蝙蝠侠》的拍摄停工。')
    print(res)
