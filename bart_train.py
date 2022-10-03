#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import unicodedata
from typing import Dict, List, Union
import numpy as np
import torch
import transformers
from datasets import DatasetDict, load_metric
from setproctitle import setproctitle
from transformers import HfArgumentParser, PreTrainedTokenizerFast, BartModel, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# Argument
from hf_bart.utils import DatasetsArguments, ModelArguments, BartTrainingArguments

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Linear+CTC loss 형태의 HuggingFace에서 지원해주는 모델의 형태입니다.
    Wav2Vec2ForCTC를 사용하도록 되어있으며, 사용할 수 있는 부분은 Auto를 최대한 활용하였습니다.
    """
    parser = HfArgumentParser((ModelArguments, DatasetsArguments, BartTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 이어서 학습 시킬 때 쓸 수 있을듯
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    train_dataset = DatasetDict()
    dev_dataset = DatasetDict()
    # clean_dataset = DatasetDict()
    # other_dataset = DatasetDict()

    # TODO: Datasets를 통합으로 만들다보니, 강제로 grapheme_labels를 사용하도록 되었다. 이후에 datasets 처리에 대해 변경이 필요하다.
    if training_args.do_train:
        train_dataset = get_concat_dataset(data_args.datasets_dirs, "train")
        train_dataset = train_dataset.rename_column("grapheme_labels", "labels")

    if training_args.do_eval:
        dev_dataset = get_concat_dataset(data_args.datasets_dirs, "dev")
        # 현재 당장 Test를 위한 KpsonSpeech만 dev로 삼음
        # dev_dataset = get_concat_dataset(["/data01/bart/temp_workspace/stt/wav2vec2_test/aihub_datasets_arrow/fine-tuning/new-data-KsponSpeech-spelling-not-normal-20"], "dev")
        dev_dataset = dev_dataset.rename_column("grapheme_labels", "labels")
        dev_dataset = dev_dataset.sort("length")

    if training_args.do_predict:
        # clean_dataset = get_concat_dataset(data_args.datasets_dirs, "eval_clean")
        # clean_dataset = clean_dataset.rename_column("grapheme_labels", "labels")
        # clean_dataset = clean_dataset.sort("length")
        # other_dataset = get_concat_dataset(data_args.datasets_dirs, "eval_other")
        # other_dataset = other_dataset.rename_column("grapheme_labels", "labels")
        # other_dataset = other_dataset.sort("length")
        # test_dataset = load_dataset("kresnik/zeroth_korean", "clean")
        datasets_dirs = ["/data2/bart/temp_workspace/stt/aihub_datasets_arrow/fine-tuning/kspon_short_eval"]
        test_dataset = get_concat_dataset(datasets_dirs, "")
        test_dataset = test_dataset.rename_column("grapheme_labels", "labels")
        test_dataset = test_dataset.sort("length")

    setproctitle(training_args.setproctitle_name)
    if is_main_process(training_args.local_rank):
        import wandb

        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.wandb_name,
        )

    # Pre-Training에서의 config과 어느정도 일맥상통하며, final_dropout과 CTC_reduction 관련 파라미터만 다르면 되므로,
    # 최대한 config를 활용하기 위해서 학습 초기에는 config를 pre-training 모델에서 가져오도록 한다.
    # predict의 경우, 이미 학습된 모델이 있다는 가정이므로, output_dir에서 가져오도록 처리.
    if training_args.do_train or training_args.do_eval:
        model_config_path = model_args.model_name_or_path
    elif training_args.do_predict:
        model_config_path = training_args.output_dir

    config = AutoConfig.from_pretrained(
        os.path.join(model_config_path, "config.json"), cache_dir=model_args.cache_dir, local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config_path,
        local_files_only=True,
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        os.path.join(model_config_path, "preprocessor_config.json"),
        cache_dir=model_args.cache_dir,
        local_files_only=True,
    )

    # gradient_checkpoint는 model config에 기본적으로 없음
    config.update(
        {
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "vocab_size": len(tokenizer),  # vocab은 뭔가 실수로 잘못 넣을 여지가 있으니 확실하게 현재 데이터에 맞게 재정의해서 사용함.
        }
    )

    # create model
    model = AutoModelForCTC.from_pretrained(
        model_config_path, cache_dir=model_args.cache_dir, config=config, local_files_only=True
    )
    model.freeze_feature_encoder()

    # Define evaluation metrics during training. 무조건 wer, cer은 측정한다.
    eval_metrics = {"eval_wer": load_metric("wer"), "eval_cer": load_metric("cer")}

    # Now save everything to be able to create a single processor later
    if training_args.do_train:
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    from pyctcdecode import build_ctcdecoder

    # lm_beam_decoder = build_ctcdecoder(
    #     labels=list(tokenizer.encoder.keys()),
    #     kenlm_model_path="/data/bart/stt/tadev-STT/transformers-wav2vec2/language_model/5gram_4data_wiki.arpa",
    #     lm_score_boundary=False
    # )
    # beamsearch_decoder = build_ctcdecoder(
    #     labels=list(tokenizer.encoder.keys()),
    #     kenlm_model_path=None,
    # )
    # from transformers import Wav2Vec2ProcessorWithLM

    # eval_loop를 진행할 시, beam_search까지 진행된 metric을 얻고 싶다면, 간단하게 하기의 펑션에서 tokenizer 부분만 수정해서 사용하면 된다.
    # processor_with_lm = Wav2Vec2ProcessorWithLM(
    #     feature_extractor=processor.feature_extractor,
    #     tokenizer=processor.tokenizer,
    #     decoder=lm_beam_decoder
    # )

    # processor_with_beam = Wav2Vec2ProcessorWithLM(
    #     feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer, decoder=beamsearch_decoder
    # )
    # processor_with_lm = Wav2Vec2ProcessorWithLM.from_pretrained(model_args.kenlm_dir)

    def compute_metrics(pred: Dict[str, Union[List[int], torch.Tensor]]) -> Dict[str, float]:
        """compute_metrics eval_loop, pred_loop에서 사용되는 compute_metrics

        argmax -> tokenizer.batch_decode: 일반적인 beam_width=1 형태의 argmax 치환 비교방식
        processor_with_lm.batch_decode: argmax를 해서 넣으면 안되며, logit으로 lm을 넣은(혹은 넣지 않은) beam_search를 계산하게 된다.

        Args:
            pred (Dict[str, Union[List[int], torch.Tensor]]): 예측값

        Returns:
            metrics (Dict[str, float]): 계산된 Metrics dict
        """
        pred_logits = pred.predictions
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        # np.set_printoptions(threshold=sys.maxsize)
        # np.savetxt('./logit_batch_1.txt', pred_logits, fmt='%10.32f', delimiter=' ', newline='\n', header='', footer='', encoding=None)
        pred_str_lm_5 = tokenizer.batch_decode(logits=pred_logits, num_processes=1, beam_width=5)
        pred_normal_lm_5 = [unicodedata.normalize("NFC", pred_text) for pred_text in pred_str_lm_5.text]
        # pred_str_lm_100 = processor_with_beam.batch_decode(logits=pred_logits, num_processes=8, beam_width=100)
        # pred_normal_lm_100 = [unicodedata.normalize("NFC", pred_text) for pred_text in pred_str_lm_100.text]
        # pred_str_beam_5 = processor_with_beam.batch_decode(logits=pred_logits, num_processes=8, beam_width=5)
        # pred_normal_beam_5 = [unicodedata.normalize("NFC", pred_text) for pred_text in pred_str_beam_5.text]
        # pred_str_beam_100 = processor_with_beam.batch_decode(logits=pred_logits, num_processes=8, beam_width=100)
        # pred_normal_beam_100 = [unicodedata.normalize("NFC", pred_text) for pred_text in pred_str_beam_100.text]

        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = tokenizer.batch_decode(pred_ids)
        pred_normal = [unicodedata.normalize("NFC", pred_text) for pred_text in pred_str]

        # with open("./100-gram_result.txt", "w") as f:
        #     for text in pred_normal_lm_100:
        #         f.write(text + "\n")
        # with open("./beam_search_result.txt", "w") as f:
        #     for text in pred_normal_beam_100:
        #         f.write(text + "\n")
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        label_normal = [unicodedata.normalize("NFC", label_text) for label_text in label_str]

        # norm_hyp_word, norm_tgt_word = get_norm_text(pred_normal_lm, label_normal)
        # err_sw, length_sw = editdistance.eval(
        #     norm_hyp_word.split(), norm_tgt_word.split()
        # ), len(norm_tgt_word.split())
        # with open("./labels.txt", "w") as f:
        #     for text in label_normal:
        #         f.write(text + "\n")
        print("Argmax")
        print({k: v.compute(predictions=pred_normal, references=label_normal) for k, v in eval_metrics.items()})
        print("beam_5")
        print({k: v.compute(predictions=pred_normal_lm_5, references=label_normal) for k, v in eval_metrics.items()})
        # print("beam_lm_5")
        # print({k: v.compute(predictions=pred_normal_lm_5, references=label_normal) for k, v in eval_metrics.items()})
        # print("beam_100")
        # print(
        # {k: v.compute(predictions=pred_normal_beam_100, references=label_normal) for k, v in eval_metrics.items()}
        # )
        # print("beam_lm_100")
        # print({k: v.compute(predictions=pred_normal_lm_100, references=label_normal) for k, v in eval_metrics.items()})
        metrics = {
            k: v.compute(predictions=pred_normal_lm_5, references=label_normal) for k, v in eval_metrics.items()
        }
        return metrics

    def preprocess_logits_for_metrics(logits, label):
        logits = logits.to("cpu") if not isinstance(logits, tuple) else logits[0].to("cpu")
        return logits

    # Instantiate custom data collator
    data_collator = DataCollatorCTCWithPadding(model=model, processor=processor)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )

    lr_scheduler = TriStageLRScheduler(
        training_args, training_args.max_steps, learning_rate=training_args.learning_rate
    )

    lr_scheduler = lr_scheduler.get_tri_stage_scheduler(optimizer)

    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=training_args.max_steps, num_cycles=training_args.num_cycles)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=dev_dataset if training_args.do_eval else None,
        tokenizer=feature_extractor,
        optimizers=(optimizer, lr_scheduler),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    if training_args.do_train:

        # use last checkpoint if exist
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(output_dir=training_args.output_dir)

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.do_predict:
        logger.info("*** Predict ***")
        print("@@@ clean_metric")
        test_dataset = test_dataset.train_test_split(test_size=0.02)
        clean_results = trainer.predict(test_dataset=test_dataset["test"])
        clean_metrics = clean_results.metrics
        clean_metrics["predict_samples"] = len(clean_metrics)

    print("@@@ other_metric")
    other_results = trainer.predict(test_dataset=other_dataset)
    other_metrics = other_results.metrics
    other_metrics["predict_samples"] = len(other_metrics)

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
