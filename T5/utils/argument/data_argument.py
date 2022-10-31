from dataclasses import dataclass, field


@dataclass
class DataArgument:
    data_name: str = field(default=None)
    num_proc: int = field(default=1)
