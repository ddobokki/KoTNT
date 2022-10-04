import os

from datasets import Dataset, load_dataset
from setproctitle import setproctitle
from transformers import HfArgumentParser, T5Config, T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from transformers.integrations import WandbCallback

from T5.data.collator import T5Collator
from T5.utils import DataArgument, ModelArgument


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    config = T5Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache
    )

    # [BUG]: datasets에서 csv를 불러올 때 무조건 columns이 명시 되어 있어야 한다.
    #        datasets은 상단의 열을 columns으로 인식하기 때문에 잘못하면 이상한 columns이 될 수 있다.
    loaded_data = load_dataset(
        "csv", data_files=data_args.data_name_or_script, cache_dir=model_args.cache, split="train"
    )

    def preprocess(input_values: Dataset) -> dict:
        """
        Text To Text: train, label 데이터도 전부 text임.
        """

        # [NOTE]: 이런 prompt의 경우 config에서 설정해야 할 듯 하다.
        #         config에 task_specific_params라는 값이 있는데 이 값을 이용하는게 HF 개발자가 의도한 사용법이 아닐까 생각함.
        # [NOTE]: 임시로 설정한 이름들, 나중에 데이터를 받아보면 삭제됨.
        train_col_name = "sentence"
        label_col_name = "label"

        train_prompt = "translation_num_to_text"
        label_prompt = "label"

        train_input = f"{train_prompt}: {input_values[train_col_name]}"
        label_input = f"{label_prompt}: {input_values[label_col_name]}"

        # [NOTE]: train이라는 이름은 나중에 바꾸는 게 좋을 듯 valid, test도 있어서 맞지가 않는다.
        train_encoded = tokenizer(train_input, return_attention_mask=False)
        label_encoded = tokenizer(label_input, return_attention_mask=False)

        result = {train_col_name: train_encoded, label_col_name: label_encoded}

        return result

    # [NOTE]: data preprocess
    desc_name = "T5_preprocess"
    loaded_data = loaded_data.map(preprocess, num_proc=data_args.num_proc, desc=desc_name)

    # [NOTE]: check data
    if train_args.do_eval or train_args.do_predict:
        splited_data = loaded_data.train_test_split(0.1)
        train_data = splited_data["train"]
        valid_data = splited_data["test"]
    else:
        train_data = loaded_data
        valid_data = None

    # [NOTE]: load collator
    collator = T5Collator(tokenizer=tokenizer)

    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collator,
        callbacks=callbacks,
    )

    if train_args.do_train:
        train(trainer, label_name="label")
    if train_args.do_eval:
        eval(trainer)
    if train_args.do_predict:
        predict(trainer)


def train(trainer: Trainer, label_name: str) -> None:
    outputs = trainer.train(ignore_keys_for_eval=label_name)
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval(trainer: Trainer) -> None:
    outputs = trainer.evaluate()
    metrics = outputs.metrics

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer: Trainer) -> None:
    outputs = trainer.predict()
    metrics = outputs.metrics

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


def set_wandb_env(process_name) -> None:
    # [NOTE]: 혹은 launch.json에서 설정하는 것도 가능.
    os.environ["WANDB_CACHE_DIR"] = "/data/jsb193/github/t5/.cache"
    os.environ["WANDB_DIR"] = "/data/jsb193/github/KoGPT_num_converter/T5/wandb"
    os.environ["WANDB_NAME"] = process_name
    os.environ["WANDB_NOTEBOOK_NAME"] = "test run t5, just ignore"  # --check name
    os.environ["WANDB_USERNAME"] = "jp_42maru"
    os.environ["WANDB_RUN_GROUP"] = "tadev"
    os.environ["WANDB_TAGS"] = "T5, finetune, test"
    os.environ["WANDB_DISABLE_CODE"] = "false"


if __name__ == "__main__":
    parser = HfArgumentParser([TrainingArguments, ModelArgument, DataArgument])
    process_name = "T5"
    setproctitle(process_name)

    os.environ["WANDB_DISABLED"] = "true"
    if os.getenv("WANDB_DISABLED") == "false":
        # [NOTE]: check wandb env variable
        # -> 환경 변수를 이용해 상세한 조작이 가능함.
        #    https://docs.wandb.ai/guides/track/advanced/environment-variables
        set_wandb_env(process_name)

    main(parser)
