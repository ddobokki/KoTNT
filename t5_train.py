import os
from typing import Any, Dict, Tuple, Union

import numpy as np
from datasets import Dataset, load_dataset
import torch
from evaluate import load
from setproctitle import setproctitle
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    set_seed,
)
from transformers.integrations import WandbCallback
from transformers.trainer_utils import EvalPrediction

from utils import DataArgument, ModelArgument, set_task_specific_params


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    # [NOTE]: 이 부분은 언제든지 수정될 수 있음. argument에서 값이 전달되지 않았을 때 애러가 발생하도록 하는 방법이 있을 거다.
    assert model_args.task is None, "Must set model task, please insert your prompt!!"

    def preprocess(input_values: Dataset) -> dict:
        """"""
        # prompt = "translation_num_to_text"
        train_input = f"""{prompt}: {input_values["num_col"]}"""
        label_input = input_values["sen_col"]

        # [NOTE]: train이라는 이름은 나중에 바꾸는 게 좋을 듯 valid, test도 있어서 맞지가 않는다.
        train_encoded = tokenizer(train_input, return_attention_mask=False, max_length=240)
        label_encoded = tokenizer(label_input, return_attention_mask=False, max_length=240)

        result = {"sen_col": train_encoded["input_ids"], "num_col": label_encoded["input_ids"]}
        return result

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """"""
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

    def logits_for_metrics(logits: Union[Tuple, torch.Tensor], _) -> torch.Tensor:
        """"""
        return_logits = logits[0].argmax(dim=-1)
        return return_logits

    # [NOTE]
    # 원래라면 model_name_or_path지만 그러니 변수의 길이가 너무 길어져서 indentation발생한다.
    # 그래서 일단 변수의 이름을 **name**으로 통일한다.

    # [NOTE]: load model, tokenizer, config
    tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name, cache_dir=model_args.cache)
    config_name = model_args.model_name if model_args.config_name is None else model_args.config_name
    config = T5Config.from_pretrained(config_name, cache_dir=model_args.cache)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name, config=config, cache_dir=model_args.cache)
    model.resize_token_embeddings(len(tokenizer))  # ??

    # [NOTE]: set default taks_specifi_params & set gen_kwargs
    config = set_task_specific_params(config) if config.task_specific_params is None else config
    task = config.task_specific_params[model_args.task]
    prompt = task.pop("prefix")
    gen_kwargs = task

    # [NOTE]: load datasets & preprocess data
    loaded_data = load_dataset("csv", data_files=data_args.data_name, cache_dir=model_args.cache, split="train")
    loaded_data = loaded_data.map(preprocess, num_proc=data_args.num_proc)
    loaded_data = loaded_data.rename_columns({"num_col": "input_ids", "sen_col": "labels"})

    if train_args.do_eval or train_args.do_predict:
        splited_data = loaded_data.train_test_split(0.08)
        train_data = splited_data["train"]
        valid_data = splited_data["test"]
    else:
        train_data = loaded_data
        valid_data = None

    # [NOTE]: load metrics & set Trainer arguments
    blue = load("evaluate-metric/bleu", cache_dir=model_args.cache)
    rouge = load("evaluate-metric/rouge", cache_dir=model_args.cache)
    collator = DataCollatorForSeq2Seq(tokenizer, model)
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        compute_metrics=metrics,
        args=train_args,
        eval_dataset=valid_data,
        data_collator=collator,
        callbacks=callbacks,
        preprocess_logits_for_metrics=logits_for_metrics,
    )

    # [NOTE]: run train, eval, predict
    if train_args.do_train:
        train(trainer, train_args)
    if train_args.do_eval:
        eval(trainer, valid_data)
    if train_args.do_predict:
        predict(trainer, valid_data, gen_kwargs)


def train(trainer: Seq2SeqTrainer, args) -> None:
    """"""
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval(trainer: Seq2SeqTrainer, eval_data: Dataset) -> None:
    """"""
    outputs = trainer.evaluate(eval_data)
    # metrics = outputs.metrics

    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    pass


def predict(trainer: Seq2SeqTrainer, test_data: Dataset, gen_kwargs: Dict[str, Any]) -> None:
    """"""
    trainer.args.predict_with_generate = True
    outputs = trainer.predict(test_data, **gen_kwargs)
    # metrics = outputs.metrics

    # trainer.log_metrics("predict", metrics)
    # trainer.save_metrics("predict", metrics)
    pass


if __name__ == "__main__":
    # example_source: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ModelArgument, DataArgument])
    # [NOTE]: check wandb env variable
    # -> 환경 변수를 이용해 조작이 가능함.
    #    https://docs.wandb.ai/guides/track/advanced/environment-variables
    main(parser)
