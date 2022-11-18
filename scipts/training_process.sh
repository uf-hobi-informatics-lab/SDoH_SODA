#training from pre-trained model on 1FL dataset
#input file should be the folder of notes files and which cuda used
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

mkdir ../models/SDOH_bert_updated_100
#mkdir ${output_dir}
#used to geneate bio file used for NER
python3 ./training_ner.py $input_dir 

# training process on 1:1 split 

python3 ../ClinicalTransformerNER/src/run_transformer_ner.py \
      --model_type bert \
      --pretrained_model ../models/ner_bert \
      --data_dir ../bio/bio_training_new \
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
      --num_train_epochs 30 \
      --gradient_accumulation_steps 1 \
      --do_warmup \
      --seed 13 \
      --warmup_ratio 0.1 \
      --max_num_checkpoints 3 \
      --log_file ../logs/log_ner_training.txt \
      --progress_bar \
      --early_stop 3 

# predict on test notes
python3 ../ClinicalTransformerNER/src/run_transformer_batch_prediction.py \
      --model_type bert \
      --pretrained_model ../models/SDOH_bert_updated_100 \
      --raw_text_dir ../temp/test_set_new_encoded \
      --preprocessed_text_dir ../bio/bio_test_new \
      --output_dir ../results/test_result \
      --max_seq_length 128 \
      --do_lower_case \
      --eval_batch_size 8 \
      --log_file ../logs/log_ner_training.txt\
      --do_format 1 \
      --do_copy \
      --data_has_offset_information

python ./brat_eval.py --f1 ../temp/test_set_new_encoded --f2 ../results/test_result_formatted_output >> ${output_dir}/eval_result_training_new.txt

