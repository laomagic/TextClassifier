**THUCNews**中文文本分类数据集的处理，该数据集包含 84 万篇新闻文档，总计 14 类；在数据集的基础上可以进行文本分类、词向量的训练等任务。数据集的下载地址：[http://thuctc.thunlp.org/](http://thuctc.thunlp.org/)

## 1.bert预训练模型的介绍

wobert/**wonezha**:以词为单位的中文 bert 模型，具体详情见：[https://github.com/ZhuiyiTechnology/WoBERT](https://github.com/ZhuiyiTechnology/WoBERT)

google/Roberta：以字为粒度的中文 bert，Roberta 的模型地址：[https://github.com/ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)，roberta 的模型为：`RoBERTa-wwm-ext, Chinese`

google 中文 bert 的地址：[https://github.com/google-research/bert](https://github.com/google-research/bert)，google 的为中文的 bert 模型

其中，四种版本的bert链接内部均有较详细的介绍。

## 2.bert模型的在数据集上测试

在THUCNews数据集上进行文本的分类任务，bert在数据集上微调过程中每 1000batch，在测试集上验证一次，待验证的loss损失在5batch内不在下降，停止训练。

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
(其中google指的goole版本的中文bert-base,如1介绍的bert对应的基础版本,本次测试的bert版本链接，百度网盘：链接：https://pan.baidu.com/s/10kX5kC20ggJo7ztz4h3rLA ；提取码：vnxk )
从测试的结果中可以看出，wobert/wonezha（词粒度版本）的bert在THUCNews文本分类的任务上的表现要远好于google/roberta（字粒度版本）,而wobert的acc比wonezha也要高1.59%。

**2）语料为字粒度版本，参数：word=False**

```plain
         acc     loss
roberta  54.01   1.4
wonezha  70.33   0.94
wobert   73.29   0.83
google   54.09   1.4
```
未使用分词工具对语料进行分词，bert默认使用的wordpiece的工具处理文本，字粒度的文本方式改变了词汇之间的相关联系，文本分类的效果总体较差，而且词粒度版本的bert比字粒度的bert的acc高16~19%，说明词粒度版本的词中一定程度上包含了char的语义信息。

## **3.FastBert蒸馏后的效果**

**FastBert的论文地址：**[https://arxiv.org/pdf/2004.02178.pdf](https://arxiv.org/pdf/2004.02178.pdf)，代码地址：[https://github.com/BitVoyage/FastBERT](https://github.com/BitVoyage/FastBERT)


### 3.1代码的使用步骤

```python
1.到数据集的地址下载原始数据进行解压，放到data目录下
运行prepare_dataset.py data文件下生成训练数据train.csv test.csv valid.csv 以及标签到id的映射文件label2id.json
运行prepare_distill_dataset.py 生成蒸馏数据train_distill_new.tsv test_distill_new.tsv dev_distill_new.tsv
2.下载预训练的bert模型 放到model目录下
3.修改utils/config.py配置文件的模型路径等参数，运行train.py文件进行文本分类任务

```

```python
模型的蒸馏，可以参考FastBert提供的代码,主要步骤：
1. 初始训练:进行文本分类的微调训练
sh run_scripts/script_train_stage0.sh
2. 蒸馏训练:transformer每层的student classifier学习teacher classifier的分布
sh run_scripts/script_train_stage1.sh
**注意**:蒸馏阶段输入数据为无监督数据，可依据需要引入更多数据提升鲁棒性
3. 推理:调节speed，观察推理速度和准确率之间的关系
sh run_scripts/script_infer.sh
其中 inference_speed参数(0.0~1.0)控制加速程度
4. 部署使用
python3 predict.py
```
### 3.2蒸馏的效果
git提供的数据集上的复现效果，使用作者提供的bert模型：
```plain
speed_arg:0.0, time_per_record:0.0365, acc:0.9392,   基准
speed_arg:0.1, time_per_record:0.0332, acc:0.9400,   1.10倍
speed_arg:0.5, time_per_record:0.0237, acc:0.9333,   1.54倍
speed_arg:0.8, time_per_record:0.0176, acc:0.9100,   2.07倍
```
推理的acc指标和git提供的结果基本一致，但是推理速度，并没有作者测试的那么好。

与2同样的数据集上，使用git提供的代码，测试代码链接：[https://github.com/laomagic/TextFastBert](https://github.com/laomagic/TextFastBert)，其中修改数据预处理文件，对train.py添加早停功能，然后进行蒸馏，蒸馏时使用的模型为wobert版本的bert，对语料使用结巴进行了分词，结果如下：

```plain
speed 0     time 0.0380  acc 0.9441  基准
speed 0.5   time 0.0364  acc 0.9441  1.04
speed 0.8   time 0.0338  acc 0.9438  1.12
speed 0.85  time 0.0267  acc 0.9432  1.42
speed 0.9   time 0.0137  acc 0.9406  2.77
speed 0.95  time 0.0080 acc 0.9229   4.75
speed 1     time 0.0062  acc 0.8974  6.13
```
未蒸馏的分类效果：acc:0.9466   loss:0.2970

未蒸馏的模型和蒸馏后的模型(speed=0)，acc相差0.0025；蒸馏后的模型，speed为0.5时，acc保持不变，而speed为0.9时，acc下降0.006,推理速度提升2.77倍，speed为1时，acc下降0.0467，推理速度提升6.14倍；表明FastBert蒸馏能保证较低精度损失的同时，一定程度能够提升模型的推理速度。