python run_language_model_roberta.py \
     --output_dir="output/roberta/" \
     --model_type="bert" \
     --model_name_or_path="roberta_ext" \
     --do_train \
     --train_data_file="wiki_train.txt" \
     --do_eval \
     --eval_data_file="wiki_eval.txt" \
     --mlm \
     --per_device_train_batch_size=8