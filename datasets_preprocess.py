from utils.DatasetsLoader import LoaderForCausalLMDatasets
import os
from transformers import HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class DatasetsArguments:
    data_dir: str = field(default="../../../data", metadata={"help": "Where is your RAW Data?"})
    is_pretraining: bool = field(
        default=False,
        metadata={"help": "Is Pre-Training? or Fine-Tuning?"},
    )
    datasets_name: str = field(default="bart", metadata={"help": "what is ForCausalLM?"})
    model_name_or_path: str = field(default=None, metadata={"help": "Where Is Your Model?"})
    num_shard: int = field(
        default=1,
    )
    output_dir: str = field(
        default="./aihub_datasets_arrow", metadata={"help": "Where is your datasets arrow output dir?"}
    )
    return_attention_mask: bool = field(
        default=False, metadata={"help": "Do you want to return not padded attention mask array?"}
    )
    datasets_generator_path: str = field(
        default="./utils/DatasetsGenerator.py",
        metadata={"help": "Where is your datasets generator?"},
    )


def main():
    parser = HfArgumentParser(DatasetsArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    print(data_args)

    if data_args.is_pretraining:
        task_path = "pre-training"
    else:
        task_path = "fine-tuning"

    loader = LoaderForCausalLMDatasets(
        data_path=data_args.data_dir,
        datasets_generator_path=data_args.datasets_generator_path,
        datasets_name=data_args.datasets_name,
        result_path=os.path.join(
            data_args.output_dir,
            task_path,
            "data-" + data_args.datasets_name + "-" + str(data_args.num_shard),
        ),
        # https://github.com/huggingface/datasets/issues/1992 issue
        # preprocessing_num_workers=8,
        overwrite_cache=False,
        model_name_or_path=data_args.model_name_or_path,
        return_attention_mask=data_args.return_attention_mask,
        data_cache_dir="./data_cache",
        train_shard_cnt=data_args.num_shard,
    )

    loader.save_preprocess_datasets(
        split=None,
    )


if __name__ == "__main__":
    main()
