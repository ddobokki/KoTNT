from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetsArguments:
    datasets_dirs: Optional[List[str]] = field(default=None)
    vocab_path: str = field(
        default=None,
    )
    feature_size: int = field(
        default=1,
    )
    padding_value: int = field(
        default=0,
    )
    do_normalize: bool = field(
        default=False,
    )
