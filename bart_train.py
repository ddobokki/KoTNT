#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import pandas as pd
from typing import Dict, List
import numpy as np
import transformers
from datasets import Dataset, load_from_disk
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    set_seed,
    BartModel,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from evaluate import load
from transformers.trainer_utils import get_last_checkpoint, is_main_process, EvalPrediction

# Argument
from utils import DatasetsArguments, ModelArguments, TNTTrainingArguments

logger = logging.getLogger(__name__)


def main() -> None:
    """
    @@@@@@@@@@@@@@@@@@@@ 파라미터 받음
    """
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, TNTTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    """
    proc name, wandb 설정
    """
    setproctitle(training_args.setproctitle_name)
    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    """
    @@@@@@@@@@@@@@@@@@@@ 이어서 학습할지 처리
    """
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    """
    @@@@@@@@@@@@@@@@@@@@ logger 세팅
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    """
    @@@@@@@@@@@@@@@@@@@@ 모델 시드 설정
    """
    set_seed(training_args.seed)

    """
    @@@@@@@@@@@@@@@@@@@@ RAW 데이터 로드
    """
    concat_list = list()
    for filename in data_args.datasets_dirs:
        df = pd.read_csv(filename, index_col=None, names=["num_sent", "ko_sent"])
        concat_list.append(df)

    df_total = pd.concat(concat_list, axis=0, ignore_index=True)
    train_datasets = Dataset.from_pandas(df_total)

    # Pre-Training에서의 config과 어느정도 일맥상통하므로, 최대한 config를 활용하기 위해서 학습 초기에는 config를 pre-training 모델에서 가져오도록 한다.
    # predict의 경우, 이미 학습된 모델이 있다는 가정이므로, output_dir에서 가져오도록 처리.
    if training_args.do_train or training_args.do_eval:
        model_path = model_args.model_name_or_path
    elif training_args.do_predict:
        model_path = training_args.output_dir

    """
    @@@@@@@@@@@@@@@@@@@@ 토크나이저 설정
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["num_sent"],
            return_length=True,
        )
        labels = tokenizer(
            batch["ko_sent"],
        )
        reversed_labels = list(reversed(labels["input_ids"]))
        reversed_labels.append(tokenizer.eos_token_id)
        model_inputs["labels"] = reversed_labels
        return model_inputs

    """
    @@@@@@@@@@@@@@@@@@@@ 토크나이징 진행, 빠르게 불러 쓰기 위해 datasets로 임시 저장
    """
    temp_datasets_dir = "/data2/bart/temp_workspace/nlp/bart_datasets/fine-tuning/bart_konuko"
    if os.path.isdir(temp_datasets_dir):
        train_datasets = load_from_disk(temp_datasets_dir)
    else:
        train_datasets = train_datasets.map(tokenize, remove_columns=["num_sent", "ko_sent"])
        train_datasets.save_to_disk(temp_datasets_dir)
    if data_args.eval_size <= 0:
        training_args.do_eval = False

    """
    @@@@@@@@@@@@@@@@@@@@ 데이터셋 분리
    """
    if training_args.do_eval:
        # do_eval과 do_predict는 동일로직을 타므로, Training 과정에서, do_eval을 안하고 do_predict만 하는 것은 의미가 없다.
        # do_train True에서는 eval만 하거나 eval과 predict을 하거나 둘중에 하나만 의미가 있다.
        train_split_datasets = train_datasets.train_test_split(test_size=data_args.eval_size, seed=training_args.seed)
        train_datasets = train_split_datasets["train"]
        dev_datasets = train_split_datasets["test"]
        if training_args.do_predict:
            dev_split_datasets = dev_datasets.train_test_split(test_size=data_args.test_size, seed=training_args.seed)
            dev_datasets = dev_split_datasets["train"]
            test_datasets = dev_split_datasets["test"]

    if not training_args.do_train and training_args.do_predict:
        # 학습하지 않고, predict만 돌리는 경우
        test_datasets = train_datasets.train_test_split(test_size=data_args.test_size, seed=training_args.seed)["test"]

    """
    @@@@@@@@@@@@@@@@@@@@ 검증 Metrics 정의
    """
    blue = load("evaluate-metric/bleu")
    rouge = load("evaluate-metric/rouge")

    def compute_metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        result = dict()

        predicts = evaluation_result.predictions
        reversed_predicts = np.flip(predicts, axis=-1)
        reversed_predicts = np.where(reversed_predicts != -100, reversed_predicts, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(reversed_predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        reversed_labels = np.flip(labels, axis=-1)
        reversed_labels = np.where(reversed_labels != -100, reversed_labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(reversed_labels, skip_special_tokens=True)

        blue_score = blue._compute(decoded_preds, decoded_labels)
        blue_score.pop("precisions")

        # latin이 아닐 시, tokenizer는 split해줘야 띄어쓰기 단위로 n-gram이 정확히 계산됨
        rouge_score = rouge._compute(decoded_preds, decoded_labels, tokenizer=lambda x: x.split())

        result.update(rouge_score)
        result.update(blue_score)

        return result

    def preprocess_logits_for_metrics(logits, labels):
        """
        @@@@@@@@@@@@@@@@@@@@ eval 전에 CPU로 빼고, --predict_with_generate=false 에 따른 argmax 처리
        """
        logits = logits.to("cpu") if not isinstance(logits, tuple) else logits[0].to("cpu")
        logits = logits.argmax(dim=-1)
        return logits

    """
    @@@@@@@@@@@@@@@@@@@@ 모델과 콜레터 선언 https://huggingface.co/course/chapter7/4?fw=pt
    """
    # encoder_model = BartModel.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_datasets if training_args.do_train else None,
        eval_dataset=dev_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    """
    @@@@@@@@@@@@@@@@@@@@ 조건의 맞는 학습, 검증, 테스트 진행
    """
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(output_dir=training_args.output_dir)

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=dev_datasets)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_results = trainer.predict(test_dataset=test_datasets)
        metrics = test_results.metrics
        metrics["predict_samples"] = len(metrics)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
