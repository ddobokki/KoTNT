from transformers import DefaultDataCollator, PreTrainedTokenizer
from typing import Any, Dict


class T5Collator(DefaultDataCollator):
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, *args: Any, **kwds: Any) -> Dict:
        return super().__call__(*args, **kwds)
