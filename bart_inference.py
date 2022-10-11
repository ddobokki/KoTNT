import os
import sys
from typing import Tuple
import torch
from transformers import HfArgumentParser, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.pipelines import Text2TextGenerationPipeline
from transformers.pipelines.text2text_generation import ReturnType

from utils import InferenceArguments


class BartText2TextGenerationPipeline(Text2TextGenerationPipeline):
    def postprocess(self, model_outputs, return_type=ReturnType.TEXT, clean_up_tokenization_spaces=False):
        records = []
        reversed_model_outputs = torch.flip(model_outputs["output_ids"][0], dims=[-1])
        for output_ids in reversed_model_outputs:
            if return_type == ReturnType.TENSORS:
                record = {f"{self.return_name}_token_ids": output_ids}
            elif return_type == ReturnType.TEXT:
                record = {
                    f"{self.return_name}_text": self.tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                    )
                }
            records.append(record)
        return records


def main(inference_args: Tuple) -> None:
    texts = ["그러게 누가 6시까지 술을 마시래?"]
    tokenizer = AutoTokenizer.from_pretrained(
        inference_args.model_name_or_path,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        inference_args.model_name_or_path,
    )

    if model.config.model_type == "bart":
        print("bart의 학습은 예측 결과가 반대로 나오므로, 결과 후처리가 필요합니다.")
        seq2seqlm_pipeline = BartText2TextGenerationPipeline(model=model, tokenizer=tokenizer)
    else:
        seq2seqlm_pipeline = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)

    kwargs = {
        "min_length": inference_args.min_length,
        "max_length": inference_args.max_length,
        "num_beams": inference_args.beam_width,
        "do_sample": inference_args.do_sample,
        "num_beam_groups": inference_args.num_beam_groups,
    }
    """
        Generates sequences of token ids for models with a language modeling head. The method supports the following
        generation methods for text-decoder, text-to-text, speech-to-text, and vision-to-text models:

            - *greedy decoding* by calling [`~generation_utils.GenerationMixin.greedy_search`] if `num_beams=1` and
              `do_sample=False`.
            - *multinomial sampling* by calling [`~generation_utils.GenerationMixin.sample`] if `num_beams=1` and
              `do_sample=True`.
            - *beam-search decoding* by calling [`~generation_utils.GenerationMixin.beam_search`] if `num_beams>1` and
              `do_sample=False`.
            - *beam-search multinomial sampling* by calling [`~generation_utils.GenerationMixin.beam_sample`] if
              `num_beams>1` and `do_sample=True`.
            - *diverse beam-search decoding* by calling [`~generation_utils.GenerationMixin.group_beam_search`], if
              `num_beams>1` and `num_beam_groups>1`.
            - *constrained beam-search decoding* by calling
              [`~generation_utils.GenerationMixin.constrained_beam_search`], if `constraints!=None` or
              `force_words_ids!=None`.
    """
    pred = seq2seqlm_pipeline(texts, **kwargs)
    print(pred)


if __name__ == "__main__":
    parser = HfArgumentParser(InferenceArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        inference_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        inference_args = parser.parse_args_into_dataclasses()[0]
    main(inference_args)
