export CUDA_VISIBLE_DEVICES=0

bz=4
epn=3
sc=2
dfmm=0
model_type=bert
pm=bert-base-uncased
data_dir=/home/zehao.yu/workspace/py3/data/dr_relation_aio_th1
nmd=./new_model
pof=./predictions.txt
log=./logs/log_1.txt

python3 ./src/relation_extraction.py \
		--model_type $model_type \
		--data_format_mode $dfmm \
		--classification_scheme $sc \
		--pretrained_model $pm \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 512 \
		--cache_data \
		--do_train \
		--do_predict \
		--do_lower_case \
		--train_batch_size $bz \
		--eval_batch_size $bz \
		--learning_rate 1e-5 \
		--num_train_epochs $epn \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \


# example of testing and convert predictions to brat
export CUDA_VISIBLE_DEVICES=1
python3 ./src/relation_extraction.py \
		--model_type $model_type \
		--data_format_mode $dfmm \
		--classification_scheme $sc \
		--pretrained_model $pm \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--overwrite_model_dir \
		--seed 13 \
		--max_seq_length 512 \
		--cache_data \
		--do_predict \
		--do_lower_case \
		--train_batch_size $bz \
		--eval_batch_size $bz \
		--learning_rate 1e-5 \
		--num_train_epochs $epn \
		--gradient_accumulation_steps 1 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 1 \
		--log_file $log \

edr="./data_annotation_entity_only"
pod="./predicted_results"
python3 src/data_processing/post_processing.py \
		--mode mul \
		--predict_result_file $pof \
		--entity_data_dir $edr \
		--test_data_file ${data_dir}/test.tsv \
		--brat_result_output_dir $pod
