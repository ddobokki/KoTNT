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
from datasets import load_metric, Dataset
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Trainer,
    set_seed,
    BartModel,
    BartForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# Argument
from hf_bart.utils import DatasetsArguments, ModelArguments, BartTrainingArguments

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Linear+CTC loss 형태의 HuggingFace에서 지원해주는 모델의 형태입니다.
    Wav2Vec2ForCTC를 사용하도록 되어있으며, 사용할 수 있는 부분은 Auto를 최대한 활용하였습니다.
    """
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, BartTrainingArguments))
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
    # https://huggingface.co/course/chapter7/6#initializing-a-new-model
    # encoder_model = BartModel.from_pretrained(model_path)
    decoder_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

    def tokenize(batch):
        outputs = tokenizer(
            batch["num_sent"],
            truncation=True,
            max_length=decoder_model.config.max_position_embeddings,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == decoder_model.config.max_position_embeddings:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = train_datasets.map(tokenize)
    tokenized_datasets
    if data_args.eval_size <= 0:
        training_args.do_eval = False

    if training_args.do_eval:
        # do_eval과 do_predict는 동일로직을 타므로, Training 과정에서, do_eval을 안하고 do_predict만 하는 것은 의미가 없다.
        # do_train True에서는 eval만 하거나 eval과 predict을 하거나 둘중에 하나만 의미가 있다.
        train_split_datasets = train_datasets.train_test_split(test_size=data_args.eval_size)
        train_datasets = train_split_datasets["train"]
        dev_datasets = train_split_datasets["test"]
        if training_args.do_predict:
            dev_split_datasets = dev_datasets.train_test_split(test_size=data_args.test_size)
            dev_datasets = dev_split_datasets["train"]
            test_datasets = dev_split_datasets["test"]

    if not training_args.do_train and training_args.do_predict:
        # 학습하지 않고, predict만 돌리는 경우
        test_datasets = train_datasets.train_test_split(test_size=data_args.test_size)["test"]

    setproctitle(training_args.setproctitle_name)
    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    # Define evaluation metrics during training. 무조건 wer, cer은 측정한다.
    eval_metrics = {"eval_wer": load_metric("wer"), "eval_cer": load_metric("cer")}

    def compute_metrics(pred: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, float]:
        """compute_metrics eval_loop, pred_loop에서 사용되는 compute_metrics

        argmax -> tokenizer.batch_decode: 일반적인 beam_width=1 형태의 argmax 치환 비교방식
        processor_with_lm.batch_decode: argmax를 해서 넣으면 안되며, logit으로 lm을 넣은(혹은 넣지 않은) beam_search를 계산하게 된다.

        Args:
            pred (Dict[str, Union[List[int], torch.Tensor]]): 예측값

        Returns:
            metrics (Dict[str, float]): 계산된 Metrics dict
        """

        pred_logits = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        pass

    def preprocess_logits_for_metrics(logits, label):
        logits = logits.to("cpu") if not isinstance(logits, tuple) else logits[0].to("cpu")
        return logits

    # Instantiate custom data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=decoder_model,
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
