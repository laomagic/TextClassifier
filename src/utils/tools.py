import logging
from logging import handlers
import re
from src.utils import config
import jieba
import json
from tqdm import tqdm
# jieba.enable_parallel(4)
import pandas as pd
import time
from functools import partial, wraps
from datetime import timedelta

tqdm.pandas()


def timethis(func=None, log=logging.getLogger(), loglevel=logging.INFO):
    '''
    计算程序运行时间的装饰器。
    '''
    if func is None:
        return partial(timethis, log=log, loglevel=loglevel)
    log.addHandler(logging.StreamHandler())
    log.setLevel(loglevel)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        log.info(f'running {func.__name__}')
        res = func(*args, **kwargs)
        end_time = time.time()
        log.info(
            f"Comparing trainings costs {end_time - start_time:.6} seconds.")
        return res

    return wrapper


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def create_logger(log_path):
    """
    日志的创建
    :param log_path:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger




def rm_stop_word(wordList):
    '''
    @description: delete stop_word
    @param {type} wordList: input data
    @return:
    list of cut word
    '''
    wordlist = [w for w in wordList if w not in config.stopWords]
    return wordlist


def query_cut(query):
    '''
    @description: word segment
    @param {type} query: input data
    @return:
    list of cut word
    '''
    return jieba.lcut(query)


def clean_symbols(text):
    """
    对特殊符号做一些处理
    """
    text = re.sub('[0-9]+', " NUM ", str(text))
    text = re.sub('[!！]+', "!", text)
    #     text = re.sub('!', '', text)
    text = re.sub('[?？]+', "?", text)
    text = re.sub("[a-zA-Z#$%&\'()*+,-./:;：<=>@，。★、…【】《》“”‘’'!'[\\]^_`{|}~]+", " OOV ", text)
    return re.sub("\s+", " ", text)


def build_dict_dataset(data):
    data["sentence"] = data['title'] + data['content']
    # data["sentence"] = data['content']
    # 去除标点符号
    data['clean_sentence'] = data['sentence'].progress_apply(clean_symbols)
    data["cut_sentence"] = data['clean_sentence'].progress_apply(query_cut)
    data['raw_words'] = data["cut_sentence"].progress_apply(lambda x: ' '.join(x))
    return data


def multiprocess_data(data, func, worker=None, word=None):
    """多进程处理数据"""
    cpu_count = multiprocessing.cpu_count() // 2  # cpu核心数
    if worker == -1 or worker > cpu_count:
        worker = cpu_count
    print('used cpu:' + format(worker))
    length = len(data)
    chunk_size = length // worker  # 切分数据
    start = 0
    end = 0
    pool = multiprocessing.Pool(processes=worker)
    result = []
    start_time = time.time()
    while end < length:
        end = start + chunk_size
        if end > length:
            end = length
        res = pool.apply_async(func, (data[start:end], word))
        start = end
        result.append(res)
    pool.close()  # 关闭进程池
    pool.join()   # 主进程等待进程结束
    end_time = time.time()
    print('multiprocess time: %.4f seconds.' % (end_time - start_time))

    # result = np.concatenate([i.get() for i in result], axis=0)
    total_data = pd.concat([i.get() for i in result], axis=0)  # 合并数据
    return total_data



if __name__ == '__main__':
    # str1 = '我的世界充3满奸诈，dfs ,的 各种,111, 222,放手'
    # st = clean_str(str1)
    # print(st)
