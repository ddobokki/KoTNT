from typing import Dict

import numpy as np
from evaluate import load
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction


class TNTEvaluator:
    def __init__(self, tokenizer:PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        self.blue = load("evaluate-metric/bleu")
        self.rouge = load("evaluate-metric/rouge")
        
    def compute_metrics(self, evaluation_result: EvalPrediction) -> Dict[str, float]:
        result = {}

        predicts = evaluation_result.predictions
        predicts = np.where(predicts != -100, predicts, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        blue_score = self.blue._compute(decoded_preds, decoded_labels)
        blue_score.pop("precisions")

        # latin이 아닐 시, tokenizer는 split해줘야 띄어쓰기 단위로 n-gram이 정확히 계산됨
        rouge_score = self.rouge._compute(decoded_preds, decoded_labels, tokenizer=lambda x: x.split())

        result.update(rouge_score)
        result.update(blue_score)

        return result
