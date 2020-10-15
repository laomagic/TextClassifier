**THUCNews**中文文本分类数据集的处理，该数据集包含 84 万篇新闻文档，总计 14 类；在数据集的基础上可以进行文本分类、词向量的训练等任务。数据集的下载地址：[http://thuctc.thunlp.org/](http://thuctc.thunlp.org/)

# 1.bert预训练模型的介绍

wobert/**wonezha**:以词为单位的中文 bert 模型，具体详情见：[https://github.com/ZhuiyiTechnology/WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)

google/Roberta：以字为粒度的中文 bert，Roberta 的模型地址：[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)，roberta 的模型为：`RoBERTa-wwm-ext, Chinese`

google 中文 bert 的地址：[https://github.com/google-research/bert](https://github.com/google-research/bert)，google 的为中文的 bert 模型

# 2.bert模型的在数据集上测试

在**THUCNews数据集上进行文本的分类任务，**bert在数据集上微调过程中每 1000batch，在测试集上验证一次，待验证的loss损失在5batch内不在下降，停止训练。

bert 测试参数设置：Bert 模型：12-layer,768-hidden,12-heads

```plain
batch_size = 64
hidden_size = 768
dropout = 0.3 
eps = 1e-8
learning_rate = 2e-5  # 学习率
```
**1）语料为词粒度版本，参数：word=True（**语料使用jieba分词工具进行分词**）**
```plain
         acc     loss
roberta  78.27   0.69
wonezha  92.24   0.25
wobert   93.83   0.2
google   77.39   0.72
```
(其中google指的goole版本的中文bert-base,如1介绍的bert对应的基础版本)

**2）语料为字粒度版本，参数：word=False**


# **3.FastBert蒸馏后的效果**

**FastBert的论文地址：**[https://arxiv.org/pdf/2004.02178.pdf](https://arxiv.org/pdf/2004.02178.pdf)，代码地址：[https://github.com/BitVoyage/FastBERT](https://github.com/BitVoyage/FastBERT)

