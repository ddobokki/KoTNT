# /bin/bash

python3 /data2/bart/temp_workspace/nlp/KoGPT_num_converter/datasets_preprocess.py \
    --data_dir=/data2/bart/temp_workspace/nlp/KoGPT_num_converter/data/raw \
    --datasets_name=bart \
    --model_name_or_path=/data2/bart/temp_workspace/nlp/models/kobart-base-v2 \
    --output_dir=/data2/bart/temp_workspace/nlp/bart_datasets \
    --datasets_generator_path=/data2/bart/temp_workspace/nlp/KoGPT_num_converter/utils/DatasetsGenerator.py