from dataclasses import dataclass, field
from utils.arguments import ModelArguments
from typing import Optional


@dataclass
class InferenceArguments(ModelArguments):
    beam_width: Optional[int] = field(default=100, metadata={"help": "BeamSearch Width. Default to 100"})
    min_length: Optional[int] = field(
        default=0,
        metadata={
            "help": "The maximum length the generated tokens can have. Corresponds to the length of the input prompt + \
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in \
                the prompt."
        },
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum length the generated tokens can have. Corresponds to the length of the input prompt + \
                `max_new_tokens`. In general, prefer the use of `max_new_tokens`, which ignores the number of tokens in \
                the prompt."
        },
    )
    do_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."},
    )
    num_beam_groups: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of groups to divide `num_beams` into in order to ensure diversity among different groups of \
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details."
        },
    )
