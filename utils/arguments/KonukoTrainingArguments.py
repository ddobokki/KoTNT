from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments


@dataclass
class KonukoTrainingArguments(Seq2SeqTrainingArguments):
    # https://github.com/huggingface/transformers/blob/v4.22.2/src/transformers/training_args_seq2seq.py#L28
    setproctitle_name: Optional[str] = field(
        default="", metadata={"help": "process name (Could see nvidia-smi process name)"}
    )
    wandb_project: Optional[str] = field(default="", metadata={"help": "wandb project name for logging"})
    wandb_entity: Optional[str] = field(
        default="", metadata={"help": "wandb entity name(your wandb (id/team name) for logging"}
    )
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb job name for logging"})
