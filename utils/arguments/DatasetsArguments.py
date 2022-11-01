from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class DatasetsArguments:
    train_csv_paths: Union[Optional[List[str]], str] = field(
        default=None,
        metadata={"help": ""},
    )
    valid_csv_paths: Union[Optional[List[str]], str] = field(
        default=None,
        metadata={"help": ""},
    )
    max_length: int = field(
        default=512,
        metadata={"help": ""},
    )
    num_proc: int = field(
        default=1,
        metadata={"help": ""},
    )
    temp_datasets_dir: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    vocab_path: str = field(
        default=None,
        metadata={"help": ""},
    )
