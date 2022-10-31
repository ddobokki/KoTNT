from typing import Any, Dict
from transformers import PretrainedConfig


# Dict[str, Dict[str:Any]]
def set_task_specific_params(config: PretrainedConfig) -> PretrainedConfig:
    """"""
    task_specific_params = {
        "task_specific_params": {
            "summarization": {
                "early_stopping": True,
                "length_penalty": 2.0,
                "max_length": 200,
                "min_length": 30,
                "no_repeat_ngram_size": 3,
                "num_beams": 4,
                "prefix": "summarize: ",
            },
            "translation_num_to_text": {
                "early_stopping": True,
                "max_length": 500,
                "num_beams": 5,
                "prefix": "translation_num_to_text: ",
            },
            "translation_text_to_num": {
                "early_stopping": True,
                "max_length": 500,
                "num_beams": 5,
                "prefix": "translation_text_to_num: ",
            },
        },
    }
    config.update(task_specific_params)
    return config
