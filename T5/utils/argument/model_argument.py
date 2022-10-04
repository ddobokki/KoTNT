from dataclasses import dataclass, field


@dataclass
class ModelArgument:
    model_name_or_path: str = field(default=None)
    cache: str = field(default=None)
    task: str = field(default="translation_digit_to_text")
