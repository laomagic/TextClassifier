from torch.utils.data import Dataset
import pandas as pd
from src.utils import config
from src.utils.tools import clean_symbols, query_cut, processes_data, multiprocess_data
from tqdm import tqdm
import torch

tqdm.pandas()


class BertDataset(Dataset):
    """数据集的创建"""

    def __init__(self, path, tokenizer=None, word=False, debug=False, max_length=128):
        super(BertDataset, self).__init__()
        self.word = word
        self.data = pd.read_csv(path, sep='\t').dropna()
        if debug:
            self.data = self.data.sample(n=1000, replace=True)
        self.data = multiprocess_data(self.data, processes_data, worker=10, word=word)  # 处理数据
        # # self.data["sentence"] = self.data['title']
        # # self.data["sentence"] = self.data['title'] + self.data['content']
        # self.data["sentence"] = self.data['content']
        # self.data['clean_sentence'] = self.data['sentence'].progress_apply(clean_symbols)
        # self.data["cut_sentence"] = self.data['clean_sentence']
        # # 标签映射到id
        # self.data['category_id'] = self.data['label'].progress_apply(lambda x: x.strip()).map(config.label2id)
        # # char粒度
        # if self.word:
        #     self.data["cut_sentence"] = self.data['clean_sentence'].progress_apply(query_cut)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, i):
        data = self.data.iloc[i]
        text = data['cut_sentence'].split(' ')  # text数据
        labels = int(data['category_id'])
        text_dict = self.tokenizer.encode_plus(
            text,
            add_special_token=True,
            max_length=self.max_length,
            ad_to_max_length=True,
            return_attention_mask=True)
        input_ids, attention_mask, token_type_ids = text_dict['input_ids'], \
                                                    text_dict['attention_mask'], \
                                                    text_dict['token_type_ids']
        output = {
            "token_ids": input_ids,
            'attention_mask': attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        return output

    def __len__(self):
        return self.data.shape[0]


def collate_fn(batch):
    """
    动态padding,返回Tensor
    :param batch:
    :return: 每个batch id和label
    """

    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])  # batch中样本的最大的长度
    labels = torch.tensor([data["labels"] for data in batch])
    token_type_ids = [data["token_type_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    # 填充每个batch的sample
    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    attention_mask_padded = padding(attention_mask, max_length)

    return token_ids_padded, attention_mask_padded, token_type_ids_padded, labels
