from dataclasses import dataclass, field


@dataclass
class DataArgument:
    data_name_or_script: str = field(default=None)
    num_proc: int = field(default=1)
