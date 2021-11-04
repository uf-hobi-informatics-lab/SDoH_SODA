#training from pre-trained model on 1FL dataset
while getopts :i:d:n:c: flag
do
    case "${flag}" in
        i) input_dir=${OPTARG};;
        c) cuda=${OPTARG};;
    esac
done
echo "Input dir: $input_dir";
echo "CUDA used: $cuda";
export CUDA_VISIBLE_DEVICES=$cuda
output_dir=../results
output_name=bio_training
mkdir ../models/SDOH_bert_updated_150
mkdir ../models/SDOH_bert_updated_100
mkdir ${output_dir}
python3 ./training_ner.py $input_dir 
python3 ../ClinicalTransformerNER/src/run_transformer_ner.py \
      --model_type bert \
      --pretrained_model ../models/ner_bert \
      --data_dir ../bio/bio_training_150 \
      --new_model_dir ../models/SDOH_bert_updated_150 \
      --overwrite_model_dir \
      --max_seq_length 128 \
      --data_has_offset_information \
      --save_model_core \
      --do_train \
      --model_selection_scoring strict-f_score-1 \
      --do_lower_case \
      --train_batch_size 8 \
      --train_steps 1000 \
      --learning_rate 1e-5 \
      --num_train_epochs 2 \
      --gradient_accumulation_steps 1 \
      --do_warmup \
      --seed 13 \
      --warmup_ratio 0.1 \
      --max_num_checkpoints 3 \
      --log_file ../logs/log_ner_training.txt \
      --progress_bar \
      --early_stop 3 


python3 ../ClinicalTransformerNER/src/run_transformer_batch_prediction.py \
      --model_type bert \
      --pretrained_model ../models/SDOH_bert_updated_150 \
      --raw_text_dir ../data/test_set_150 \
      --preprocessed_text_dir ../bio/bio_test_150 \
      --output_dir ../result/training_result_150 \
      --max_seq_length 128 \
      --do_lower_case \
      --eval_batch_size 8 \
      --log_file ../logs/log_ner_training.txt\
      --do_format 1 \
      --do_copy \
      --data_has_offset_information

python ./brat_eval.py --f1 ../data/test_set_150 --f2 ../result/training_result_150_formatted_output >> ${output_dir}/eval_result_training_150.txt


# training process on 1:1 split 

python3 ../ClinicalTransformerNER/src/run_transformer_ner.py \
      --model_type bert \
      --pretrained_model ../models/ner_bert \
      --data_dir ../bio/bio_training_100 \
      --new_model_dir ../models/SDOH_bert_updated_100 \
      --overwrite_model_dir \
      --max_seq_length 128 \
      --data_has_offset_information \
      --save_model_core \
      --do_train \
      --model_selection_scoring strict-f_score-1 \
      --do_lower_case \
      --train_batch_size 8 \
      --train_steps 1000 \
      --learning_rate 1e-5 \
      --num_train_epochs 2 \
      --gradient_accumulation_steps 1 \
      --do_warmup \
      --seed 13 \
      --warmup_ratio 0.1 \
      --max_num_checkpoints 3 \
      --log_file ../logs/log_ner_training.txt \
      --progress_bar \
      --early_stop 3 


python3 ../ClinicalTransformerNER/src/run_transformer_batch_prediction.py \
      --model_type bert \
      --pretrained_model ../models/SDOH_bert_updated_100 \
      --raw_text_dir ../data/test_set_100 \
      --preprocessed_text_dir ../bio/bio_test_100 \
      --output_dir ../result/training_result_100 \
      --max_seq_length 128 \
      --do_lower_case \
      --eval_batch_size 8 \
      --log_file ../logs/log_ner_training.txt\
      --do_format 1 \
      --do_copy \
      --data_has_offset_information

python ./brat_eval.py --f1 ../data/test_set --f2 ../result/training_result_100_formatted_output >> ${output_dir}/eval_result_training_100.txt

