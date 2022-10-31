from dataclasses import dataclass, field


@dataclass
class ModelArgument:
    model_name: str = field(default=None)
    config_name: str = field(default=None)
    cache: str = field(default=None)
    task: str = field(default=None)
