import time
from src.utils import config
from src.utils.tools import create_logger
from importlib import import_module
import argparse
import numpy as np
import torch
from sklearn import metrics
import torch.nn.functional as F
from tqdm import tqdm
from src.bert_dataset import BertDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForSequenceClassification, BertTokenizer,
                          RobertaConfig, RobertaForSequenceClassification,
                          RobertaTokenizer,
                          XLNetConfig, XLNetForSequenceClassification,
                          XLNetTokenizer, get_linear_schedule_with_warmup)

parse = argparse.ArgumentParser(description='文本分类')
parse.add_argument('--model', type=str, default='BERT', help='选择模型: CNN, RNN, RCNN, RNN_Att, DPCNN, Transformer')
parse.add_argument('--word', default=True, type=bool, help='词或者字符')
parse.add_argument('--dictionary', default=config.dict_path, type=str, help='字典的路径')
args = parse.parse_args()


def train(config, model, train_iter, dev_iter, test_iter, model_name):
    start_time = time.time()
    model.train()
    print('User AdamW...')
    print(config.device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      eps=config.eps)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, mask, tokens, labels) in tqdm(enumerate(train_iter)):
            trains = trains.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            outputs = model((trains, mask, tokens))
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            #             scheduler.step()
            if total_batch % 1000 == 0 and total_batch != 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = round(time.time() - start_time, 4)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(
                    msg.format(total_batch, loss.item(), train_acc, dev_loss,
                               dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config,
                                                                model,
                                                                test_iter,
                                                                test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = round(time.time() - start_time, 4)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, mask, tokens, labels in tqdm(data_iter):
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            mask = mask.to(config.device)
            tokens = tokens.to(config.device)
            outputs = model((texts, mask, tokens))
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all,
                                               predict_all,
                                               target_names=config.label_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


if __name__ == '__main__':
    model_name = args.model
    x = import_module('models.' + model_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    logger = create_logger(config.root_path + '/logs/train.log')

    logger.info('Building tokenizer')
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    logger.info('Loading dataset')
    # 数据集的定义
    train_dataset = BertDataset(config.train_path, tokenizer=tokenizer, word=args.word)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=collate_fn,
                                  shuffle=True)
    dev_dataset = BertDataset(config.valid_path, tokenizer=tokenizer, word=args.word)
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                collate_fn=collate_fn,
                                shuffle=True)
    test_dataset = BertDataset(config.test_path, tokenizer=tokenizer, word=args.word)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=collate_fn)
    logger.info('load network')
    model = x.Model(config).to(config.device)
    # 初始化参数

    logger.info('training model')
    train(config, model, train_dataloader, dev_dataloader, test_dataloader, model_name)
    # test(config, model, test_dataloader)  # 只测试模型的效果
