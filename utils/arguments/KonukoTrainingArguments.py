from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class KonukoTrainingArguments(TrainingArguments):
    setproctitle_name: Optional[str] = field(
        default="", metadata={"help": "process name (Could see nvidia-smi process name)"}
    )
    wandb_project: Optional[str] = field(default="", metadata={"help": "wandb project name for logging"})
    wandb_entity: Optional[str] = field(
        default="", metadata={"help": "wandb entity name(your wandb (id/team name) for logging"}
    )
    wandb_name: Optional[str] = field(default="", metadata={"help": "wandb job name for logging"})
