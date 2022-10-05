#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import pandas as pd
from typing import Dict, List, Union
import numpy as np
import torch
import transformers
import datasets
from datasets import load_metric, Dataset, load_from_disk
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
from utils import DatasetsArguments, ModelArguments, KonukoTrainingArguments

logger = logging.getLogger(__name__)


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, KonukoTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 이어서 학습 시킬 때 쓸 수 있을듯
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
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

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

    # create model
    # https://huggingface.co/course/chapter7/4?fw=pt
    # encoder_model = BartModel.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    def tokenize(batch):
        model_inputs = tokenizer(
            batch["num_sent"],
            return_length=True,
        )
        labels = tokenizer(
            batch["ko_sent"],
        )
        model_inputs["labels"] = list(reversed(labels["input_ids"]))
        return model_inputs

    temp_datasets_dir = "/data2/bart/temp_workspace/nlp/bart_datasets/fine-tuning/bart_konuko"
    if os.path.isdir(temp_datasets_dir):
        train_datasets = load_from_disk(temp_datasets_dir)
    else:
        train_datasets = train_datasets.map(tokenize, remove_columns=["num_sent", "ko_sent"])
    if data_args.eval_size <= 0:
        training_args.do_eval = False

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

    setproctitle(training_args.setproctitle_name)
    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    blue = load("evaluate-metric/bleu")
    rouge = load("evaluate-metric/rouge")

    def compute_metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        result = dict()

        predicts = evaluation_result.predictions
        predicts = np.where(predicts != -100, predicts, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        blue_score = blue._compute(decoded_preds, decoded_labels)
        blue_score.pop("precisions")

        rouge_score = rouge._compute(decoded_preds, decoded_labels)

        result.update(rouge_score)
        result.update(blue_score)

        return result

    def preprocess_logits_for_metrics(logits, label):
        logits = logits.to("cpu") if not isinstance(logits, tuple) else logits[0].to("cpu")
        logits = logits[0].argmax(dim=-1)
        return logits

    # Instantiate custom data collator
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

    # Training
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
