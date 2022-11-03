from typing import Any, Dict
from transformers import PretrainedConfig

#################################
# 필요 없다고 생각하면 지워주세요 #
#################################


def set_task_specific_params(config: PretrainedConfig) -> PretrainedConfig:
    """_set_task_specific_params_
        predict시 model.generate의 BeamSearch를 위한 값을 설정하는 함수입니다.
        이 값들은 prefix만 제외한 나머지 값들은 kwargs값으로 model.generate에 전달됩니다.
        prefix는 기본적으로 key값과 동일해야 합니다.

        - huggingface.PretrainedConfig.task_specific_params
        https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig.task_specific_params

        T5마다 task_specific이 설정되지 않은 경우가 많기 때문에 설정하기 위해 만든 합수 입니다.
    Args:
        config (PretrainedConfig): 값을 넣어줄 config를 전달받습니다.

    Returns:
        PretrainedConfig: task_specific_params값을 넣은 config가 반환됩니다.
    """

    task_specific_params = {
        "task_specific_params": {
            "translation_num_to_txt": {
                "early_stopping": True,
                "max_length": 500,
                "num_beams": 5,
                "prefix": "translation_num_to_txt: ",
            },
            "translation_txt_to_num": {
                "early_stopping": True,
                "max_length": 500,
                "num_beams": 5,
                "prefix": "translation_txt_to_num: ",
            },
        },
    }
    config.update(task_specific_params)
    return config
