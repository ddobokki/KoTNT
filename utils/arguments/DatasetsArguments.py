from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetsArguments:
    train_csv_paths: Optional[List[str]] = field(default=None)
    valid_csv_paths: Optional[List[str]] = field(default=None)
    temp_datasets_dir: Optional[str] = field(default=None)
    vocab_path: str = field(
        default=None,
    )
