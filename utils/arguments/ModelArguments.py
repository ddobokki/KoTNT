from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    direction: Optional[str] = field(
        default="forward",
        metadata={"help": "How inference your label direction? default: forward [forward, backward]"},
    )
    t5_task: str = field(
        default=None,
        metadata={"help": ""},
    )
    config_name_or_path: str = field(
        default=None,
        metadata={"help": ""},
    )
