# make trn file for make script
# ksponspeech는 trn 파일을 가지고 있으며, 해당 구성 세트로 make vocab등의 작업이 수반된다.
# 다른 회사에서 만든 데이터 세트는 ETRI 전사규칙을 따르더라도, 폴더구조, trn파일의 유무는 같지 않다.
# 때문에 회사별 폴더구조 규칙등을 고려하여, trn파일을 만들 수 있도록 처리한다.

DATA_ROOT_DIR=../data/raw/
IS_TRAIN=1
IS_MAKE_DEV=1
SCENARIO_NAME=KconfSpeech

python ../../raw_transcripts_preproc/make_aihub_to_trn.py \
    --data_root_dir ${DATA_ROOT_DIR} \
    --output_dir /data2/bart/temp_workspace/nlp/KoGPT_num_converter/data/raw \
    --scenario_name ${SCENARIO_NAME} \
    --is_train ${IS_TRAIN} \
    --is_make_dev ${IS_MAKE_DEV}
