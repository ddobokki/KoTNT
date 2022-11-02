import logging
import os
import sys
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd
import transformers
from datasets import load_dataset
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# Argument
from utils import DatasetsArguments, ModelArguments, TNTTrainingArguments
from utils.metrics import TNTEvaluator, preprocess_logits_for_metrics
from utils.preprocess_func import bart_preprocess, gpt_preprocess, t5_preprocess

logger = logging.getLogger(__name__)
DATA_EXTENTION = "csv"


def main(model_args: ModelArguments, data_args: DatasetsArguments, training_args: TNTTrainingArguments):
    set_seed(training_args.seed)

    data_files = {"train": data_args.train_csv_paths, "validation": data_args.valid_csv_paths}
    dataset = load_dataset(DATA_EXTENTION, data_files=data_files, cache_dir=model_args.cache_dir)
    # [TODO] valid가 없으면 train에서 스플릿해서 나누는 코드 작성
    # train_csv_paths? s?

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_max_length = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else config.n_ctx

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=model_max_length,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    if "gpt" in model_args.model_name_or_path:
        preprocess = partial(gpt_preprocess, tokenizer=tokenizer, train_type=training_args.train_type)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
    elif "t5" in model_args.model_name_or_path:
        preprocess = t5_preprocess
    elif "bart" in model_args.model_name_or_path:
        preprocess = bart_preprocess

    dataset = dataset.map(preprocess, num_proc=data_args.num_proc, remove_columns=dataset["train"].column_names)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    # [TODO] 완성하기
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    # print("in")

    tnt_metrics = TNTEvaluator(tokenizer=tokenizer).compute_metrics
    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

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

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=tnt_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=valid_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

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
        metrics = trainer.evaluate(eval_dataset=valid_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        test_results = trainer.predict(test_dataset=valid_dataset)
        metrics = test_results.metrics
        metrics["predict_samples"] = len(metrics)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, TNTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    main(model_args=model_args, data_args=data_args, training_args=training_args)
