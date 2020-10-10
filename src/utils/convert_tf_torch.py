import argparse
import torch
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert, BertModel, BertTokenizer
import logging


logger = logging.getLogger('convert_tf_torch')


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    logger.info("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    logger.info("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


    



if __name__ == "__main__":
    tf_checkpoint_path = './chinese_wonezha_L-12_H-768_A-12/bert_model.ckpt'  # tf模型文件
    bert_config_file = './chinese_wonezha_L-12_H-768_A-12/bert_config.json'  # tf模型配置
    pytorch_dump_path = './wonezha_bert/pytorch_model.bin'  # 转换后的模型文件，配置文件和词典和tf的一致
    convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path)
    # 测试转换后的模型
    model = BertModel.from_pretrained('./wonezha_bert/')
    logger.info('loading bert model')
    print(model)
    tokenizer = BertTokenizer.from_pretrained('./wonezha_bert/')
    text_dict = tokenizer.encode_plus('自然语言处理',return_tensors='pt')
    input_ids = text_dict['input_ids']
    token_type_ids = text_dict['token_type_ids']
    attention_mask= text_dict['attention_mask']
    res  = model(input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask)
    print(res)