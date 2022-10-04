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
    eval_size: float = field(default=0.0, metadata={"help": "전체 중에서 eval로 쓸 비율"})
    test_size: float = field(
        default=0.0, metadata={"help": "do_train: eval중에 test로 쓸 비율, not do_train: 전체 중에서 test로 쓸 비율"}
    )
