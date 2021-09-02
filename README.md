# Requirement
- python env: 3.8+
- use ```pip install -r requirements.txt``` to install dependencies


# Models
- we have two models trained for NER and Relation
- Both models base on BERT architecture with different classifiers
- we provide models based on request
- contact: zehao.yu@ufl.edu; alexgre@ufl.edu; yonghui.wu@ufl.edu


# SDoH_NLPend2end System
- The system aims for extract SDoH information from clinical notes
- We support text format for production and brat format for evaluation
- The system is a two stage pipeline
  - The first stage is to extract SDoH concepts
  - The second stage is to identify relations between extracted concepts
  

# Usage
- download the models and unzip into this directory you should have:
    - ./models/ner_bert
    - ./models/re_bert
- cd to the ```./scripts``` directory
- execute pipeline as 
```shell
bash run_pred.sh -i <input data directory> -c gpu_id
```
- "input data directory" is the location of the data you annotated (*.txt and *.ann) e.g., ./test_data
- gpu_id is the id where you want to run the program. e.g, 0 - use the GPU with id as 0
- if GPU is not available, try -1 to use CPU which is slow but should work.


# Results
- in the main directory (./SDoH_NLPend2end), we will create three directories for outputs
- the first is ./logs which saves all the running logs
- the second is ./temp which saves all the intermediate generated files
- the third is ./results where the eval_results.txt is the final performance measurement, the rest directories are the e2e outputs