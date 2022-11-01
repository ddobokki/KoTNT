import os

# for type annotation
from argparse import Namespace
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
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

from utils import DatasetsArguments, ModelArguments, set_task_specific_params


def main(parser: HfArgumentParser) -> None:
    """_main_
        학습이 시작되는 함수입니다. 이 함수는 크게
        1. argument를 세팅
        2. model, tokenizer, config를 로드
        3. dataset 로드
        4. dataset 전처리
        5. metrics 로드
        6. collator 로드
        7. callback 로드
        8. Trainer 세팅
        9. 학습 시작
        의 과정으로 구성되어 있습니다.
    Args:
        parser (HfArgumentParser): paser값을 전달받습니다.

    """
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)
    set_seed(train_args.seed)

    assert model_args.t5_task is not None, "Must set model task, please insert your prompt!!"

    def preprocess(input_values: Dataset) -> dict:
        """_preprocess_
            순수 음절 문자열을 tokenizer를 이용해 정수로 바꾸는 함수 입니다.
            이 함수는 datasets의 map 메소드로 부터 불러온 뒤 MultiProcessing을 이용해 처리됩니다.

        Args:
            input_values (Dataset): MuliProcessing으로 부터 건내받은 Dataset을 건내받습니다.

        Returns:
            dict: dict값을 반환하며 dataset을 구성하는 columns과 동일한 이름의 key값이 반환됩니다
                  만약 다른 이름의 key값이 들어가면 datasets에서 append됩니다.
        """

        # prompt = "translation_num_to_text"
        train_input = f"""{prompt}: {input_values["num_col"]}"""
        label_input = input_values["sen_col"]

        # [NOTE]: Tokenizer에서 EOS토큰을 자동으로 붙여준다.
        train_encoded = tokenizer(train_input, return_attention_mask=False, max_length=data_args.max_length)
        label_encoded = tokenizer(label_input, return_attention_mask=False, max_length=data_args.max_length)

        train_encoded["input_ids"] = train_encoded["input_ids"][:-1]  # </eos> 재거

        result = {"input_ids": train_encoded["input_ids"], "labels": label_encoded["input_ids"]}
        return result

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        """_metrics_
            evaluation과정에서 모델의 성능을 측정하기 위한 metric을 수행하는 함수 입니다.
            이 함수는 Trainer에 의해 실행되며 Huggingface의 Evaluate 페키로 부터
            각종 metric을 전달받아 계산한 뒤 결과를 반환합니다.

        Args:
            evaluation_result (EvalPrediction): Trainer.evaluation_loop에서 model을 통해 계산된
            logits과 label을 전달받습니다.

        Returns:
            Dict[str, float]: metrics 계산결과를 dict로 반환합니다.
        """

        result = dict()

        predicts = evaluation_result.predictions
        predicts = np.where(predicts != -100, predicts, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu_score = bleu._compute(decoded_preds, decoded_labels)
        bleu_score.pop("precisions")

        rouge_score = rouge._compute(decoded_preds, decoded_labels, tokenizer=rouge_tokenizer)

        result.update(rouge_score)
        result.update(bleu_score)

        return result

    def logits_for_metrics(logits: Union[Tuple, torch.Tensor], _) -> torch.Tensor:
        """_logits_for_metrics_
            Trainer.evaluation_loop에서 사용되는 함수로 logits를 argmax를 이용해
            축소 시켜 공간복잡도를 줄이기 위한 목적으로 작성되었습니다.

        Args:
            logits (Union[Tuple, torch.Tensor]): Model을 거쳐서 나온 3차원 (bch, sqr, hdn)의 logits을 전달받습니다.
            _ : label이 입력되는 부분이지만 사용되지 않기에 하이픈처리 했습니다.

        Returns:
            torch.Tensor: 차원을 축소한 뒤의 torch.Tensor를 반환합니다.
        """
        return_logits = logits[0].argmax(dim=-1)
        return return_logits

    # [NOTE]: load model, tokenizer, config
    tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)

    check_config_type = model_args.config_name_or_path is None
    config_name = model_args.model_name_or_path if check_config_type else model_args.config_name_or_path
    config = T5Config.from_pretrained(config_name, cache_dir=model_args.cache_dir)

    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.resize_token_embeddings(len(tokenizer))

    # [NOTE]: set default taks_specifi_params & set gen_kwargs
    if train_args.do_predict:
        config = set_task_specific_params(config) if config.task_specific_params is None else config
        task = config.task_specific_params[model_args.task]
        prompt = task.pop("prefix")
        gen_kwargs = task
    else:
        prompt = model_args.t5_task

    # [NOTE]: load datasets & preprocess data
    data_files = dict()
    if train_args.do_train:
        train_csv = data_args.train_csv_paths
        train_csv = train_csv if isinstance(train_csv, list) else [train_csv]
        data_files.update({"train": train_csv})

    if train_args.do_predict or train_args.do_eval:
        valid_csv = data_args.valid_csv_paths
        valid_csv = valid_csv if isinstance(valid_csv, list) else [valid_csv]
        data_files.update({"valid": valid_csv})

    assert data_files is not {}, "please set args do_train, do_eval, do_predict!!!!!!!!"
    loaded_data = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)

    train_data = loaded_data["train"].map(preprocess, num_proc=data_args.num_proc) if "train" in loaded_data else None
    valid_data = loaded_data["valid"].map(preprocess, num_proc=data_args.num_proc) if "valid" in loaded_data else None

    # [NOTE]: load metrics & set Trainer arguments
    bleu = load("evaluate-metric/bleu", cache_dir=model_args.cache_dir)
    rouge = load("evaluate-metric/rouge", cache_dir=model_args.cache_dir)
    rouge_tokenizer: str = lambda sentence: sentence.split()

    collator = DataCollatorForSeq2Seq(tokenizer, model)
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=train_args,
        compute_metrics=metrics,
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


def train(trainer: Seq2SeqTrainer, args: Namespace) -> None:
    """_train_
        Trainer를 전달받아 Trainer.train을 실행시키는 함수입니다.
        학습이 끝난 이후 학습 결과 그리고 최종 모델을 저장하는 기능도 합니다.

        만약 학습을 특정 시점에 재시작 하고 싶다면 Seq2SeqTrainingArgument의
        resume_from_checkpoint을 True혹은 PathLike한 값을 넣어주세요.

        - huggingface.trainer.checkpoint
        https://huggingface.co/docs/transformers/main_classes/trainer#checkpoints

    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        args (Namespace): Seq2SeqTrainingArgument를 전달받습니다.
    """
    outputs = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_model(args.output_dir)


def eval(trainer: Seq2SeqTrainer, eval_data: Dataset) -> None:
    """_eval_
        Trainer를 전달받아 Trainer.eval을 실행시키는 함수입니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        eval_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
    """
    trainer.evaluate(eval_data)


def predict(trainer: Seq2SeqTrainer, test_data: Dataset, gen_kwargs: Dict[str, Any]) -> None:
    """_predict_
        Trainer를 전달받아 Trainer.predict을 실행시키는 함수입니다.
        이때 Seq2SeqTrainer의 Predict이 실행되며 model.generator를 실행시키기 위해
        arg값의 predict_with_generater값을 강제로 True로 변환시킵니다.

        True로 변환시키면 model.generator에서 BeamSearch를 진행해 더 질이 좋은 결과물을 만들 수 있습니다.
    Args:
        trainer (Seq2SeqTrainer): Huggingface의 torch Seq2SeqTrainer를 전달받습니다.
        test_data (Dataset): 검증을 하기 위한 Data를 전달받습니다.
        gen_kwargs (Dict[str, Any]): model.generator를 위한 값들을 전달받습니다.
    """
    trainer.args.predict_with_generate = True
    trainer.predict(test_data, **gen_kwargs)


if __name__ == "__main__":
    # example_source: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ModelArguments, DatasetsArguments])
    # [NOTE]: check wandb env variable
    # -> 환경 변수를 이용해 조작이 가능함.
    #    https://docs.wandb.ai/guides/track/advanced/environment-variables

    main(parser)
