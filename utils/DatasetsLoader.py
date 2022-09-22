import os
from typing import List
import json
import datasets
import numpy as np
from transformers import BartTokenizerFast
from utils.sent_filters import sentence_filter


class LoaderForCausalLMDatasets:
    def __init__(
        self,
        datasets_generator_path: str = "./aihub_stt_open_datasets.py",
        datasets_name: str = "bart",
        data_path: str = "./data",
        result_path: str = "./aihub_datasets_arrow",
        data_cache_dir: str = "./.data_cache",
        model_cache_dir: str = "./.model_cache",
        model_name_or_path: str = None,
        return_attention_mask: bool = False,
        preprocessing_num_workers: int = None,
        overwrite_cache: bool = False,
        train_shard_cnt: int = 1,
    ) -> None:
        """AihubSttDatasets.__init__()

            Aihub의 ETRI 전사 기준 STT Datasets를,
                1. load_dataset 하고
                2. map을 통한 dataset preprocess를 실시합니다.

            해당 프로세스는 온전히 wav2vec 2.0 모델을 학습하기 위함이며,
            fairseq 알고리즘과는 최대한 비슷하게 개발하였으나 구조상 약간은 상이한 부분들이 존재합니다. (우측이 Huggingface)
                1. 평균분산정규화 엡실론 수치 (1e-5 vs 1e-7)
                2. 패딩의 활용 (pad를 사용함 vs random position min max lenth croping)
                3. 데이터 길이 필터링 (pad를 사용하므로 데이터 길이를 필터링할 필요가 있음 vs 기준만큼 잘라서 활용하므로, 짧고 긴 음성들은 알아서 짤려나감)
            이외의 것들은 전부 output 동일한 것을 확인하였음.

        Args:
            datasets_generator_path (str, optional): datasets를 구성하는 python 경로. Defaults to "./data".
            data_path (str, optional): datasets를 구성하기 위한, data가 존재하는 경로. Defaults to "./data".
            result_path (str, optional): arrow dataset이 저장될 폴더 경로. Defaults to "./aihub_datasets_arrow".
            data_cache_dir (str, optional): 데이터 기준 캐시 폴더 경로. Defaults to "./.data_cache".
            model_cache_dir (str, optional): 모델 기준 캐시 폴더 경로. Defaults to "./.model_cache".
            return_attention_mask (bool, optional): 어텐션 마스크 value를 datasets에 추가할 것인지. Defaults to False.
            preprocessing_num_workers (int, optional): 멀티프로세스 사용 시 사용할 코어 개수. Defaults to None.
            overwrite_cache (bool, optional): 캐시를 다시 쓸 것인지 flag. Defaults to False.
            grapheme_vocab_path (str, optional): 음소의 vocab 경로
            syllabel_vocab_path (str, optional): 음절의 vocab 경로
            train_shard_cnt (int, optional): datasets가 큰 경우, trainer에서 I/O 부하가 커져서, 해결방안으로 제시된 shard. Defaults to 10
                참고: https://github.com/huggingface/datasets/issues/2252
        """
        self.datasets_generator_path = datasets_generator_path
        self.datasets_name = datasets_name
        self.data_path = data_path
        self.result_path = result_path

        self.data_cache_dir = data_cache_dir
        self.model_cache_dir = model_cache_dir

        self.return_attention_mask = return_attention_mask
        print("You can use num_workers max cnt:", os.cpu_count())
        self.preprocessing_num_workers = preprocessing_num_workers
        self.overwrite_cache = overwrite_cache

        self.train_shard_cnt = train_shard_cnt

        self.tokenizer = BartTokenizerFast.from_pretrained(model_name_or_path)

    def __vaild_filter(self, batch: datasets) -> datasets:
        return 2 * len(batch["feature_text"]) >= len(batch["label_text"])

    def __normalize(self, batch: datasets) -> datasets:
        """__normalize transformers.Wav2vecFeatureExtractor class를 활용하여, 정규화 실시

        fairseq datasets의 postprocess 메소드와 동일하게, sampling_rate를 비교하고, array shape을 처리한 후, 평균-분산 정규화 실시
        다만, fairseq은 torch의 layer_norm을 활용하고, huggingface는 직접 계산하는데, 이 부분에서 eps값이 다르게 처리됨.

        Args:
            batch (datasets): map의 대상 datasets

        Returns:
            datasets: 정규화 처리가 끝난 datasets
        """
        if self.tokenizer is not None:
            batch["feature"] = self.tokenizer(batch["feature_text"], return_attention_mask=self.return_attention_mask)
            batch["labels"] = self.tokenizer(batch["label_text"], return_attention_mask=self.return_attention_mask)
        return batch

    def __get_length(self, batch: datasets) -> datasets:
        """__get_audio_length 오디오의 길이를 계산합니다. 매번 계산해서 처리하는데에, 시간이 오래걸려 Datasets로 만들어둡니다.

        Args:
            batch (datasets): map의 대상 datasets

        Returns:
            datasets: 길이 컬럼("length") 추가가 끝난 datasets
        """
        batch["length"] = len(batch["feature"]["input_ids"])
        return batch

    def __preprocess_datasets(
        self,
        tot_datasets,
    ):
        preproc_datasets = tot_datasets.filter(
            self.__vaild_filter,
            num_proc=self.preprocessing_num_workers,
        )
        preproc_datasets = preproc_datasets.map(
            self.__normalize,
            num_proc=self.preprocessing_num_workers,
            load_from_cache_file=not self.overwrite_cache,
            remove_columns=["feature_text", "label_text"],
        )

        preproc_datasets = preproc_datasets.map(
            self.__get_length,
            num_proc=self.preprocessing_num_workers,
        )

        return preproc_datasets

    def save_preprocess_datasets(
        self,
        split: str = None,
    ) -> datasets:
        """make_preprocess_datasets datasets에 대한 전처리를 수행

        1. datasets load
        2. datasets에 지정된 path에서 file load -> float 변환
        3. 묵음 처리 (self.do_silence: True)
        4. 길이 필터링
        5. 정규화 (self.do_normalize: True)
        6. arrow dataset으로 저장
        순으로 진행됨

        Args:
            split (str, optional): 특정 나누고 싶은 데이터 셋 (train, eval, dev, eval_clean, eval_other). Defaults to None.
            min_duration_in_seconds (int, optional): 필터링할 최소 음성 길이 (초). Defaults to 1.
            max_duration_in_seconds (int, optional): 필터링할 최대 음성 길이 (초). Defaults to 20.
            top_db (int, optional): 묵음 처리하고자 하는 묵음 임계치 db. Defaults to 30.
            test_case_list (List[str], optional): 테스트 하고 싶은 구간 리스트 (['raw_audio_float','del_silence','normalize']). Defaults to None.

        Returns:
            datasets: flag에 따라 다르겠지만, 모두 True라면 묵음과 정규화가 완료된 dataset
        """

        total_datasets = datasets.load_dataset(
            path=self.datasets_generator_path,
            name=self.datasets_name,
            data_dir=self.data_path,
            split=split,
            cache_dir=self.data_cache_dir,
        )

        split_names = list(total_datasets.keys())
        for split_name in split_names:
            print(split_name + " START!!!!!!")
            if split_name == "train":
                shard_cnt = self.train_shard_cnt
            else:
                shard_cnt = 1

            for shard_idx in range(shard_cnt):
                shard_datasets = total_datasets[split_name].shard(num_shards=shard_cnt, index=shard_idx)
                shard_datasets = self.__preprocess_datasets(shard_datasets)
                shard_datasets.save_to_disk(os.path.join(self.result_path, split_name, str(shard_idx)))

        # 따로따로 처리해서 저장하는 것에 대한, huggingface datasets save_to_disk output과 동일시 하기 위해 파일을 만들어줌
        with open(os.path.join(self.result_path, "dataset_dict.json"), "w") as outfile:
            json.dump({"splits": split_names}, outfile)
