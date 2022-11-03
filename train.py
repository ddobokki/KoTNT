import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import transformers
from datasets import Dataset, load_from_disk
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    set_seed,
)
from transformers.trainer_utils import (
    EvalPrediction,
    get_last_checkpoint,
    is_main_process,
)

# Argument
from utils import DatasetsArguments, ModelArguments, TNTTrainingArguments
from utils.metrics import TNTEvaluator, preprocess_logits_for_metrics
from utils.preprocess_func import bart_preprocess, gpt_preprocess, t5_preprocess

logger = logging.getLogger(__name__)

def main(model_args:ModelArguments,data_args:DatasetsArguments,training_args:TNTTrainingArguments):
    set_seed(training_args.seed)

    if 'gpt' in model_args.model_name_or_path:
        preprocess = gpt_preprocess
    elif 't5' in model_args.model_name_or_path:
        preprocess = t5_preprocess
    elif 'bart' in model_args.model_name_or_path:
        preprocess = bart_preprocess

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    #[TODO] 완성하기
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    tnt_metrics = TNTEvaluator(tokenizer=tokenizer).compute_metrics

    trainer = Seq2SeqTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=tnt_metrics,
        # train_dataset=train_datasets if training_args.do_train else None,
        # eval_dataset=dev_datasets if training_args.do_eval else None,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )


    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    pass


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, TNTTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
