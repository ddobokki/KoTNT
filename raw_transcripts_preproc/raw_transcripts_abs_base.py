from typing import List, DefaultDict, Dict, Union
from pathlib import Path
from tqdm import tqdm
from random import shuffle
from collections import defaultdict


class RawTranscriptsAbsBase:
    def __init__(self) -> None:
        pass

    def concat_scripts(self, scripts: List[str]) -> str:
        """
        리스트의 Ksponspeech 형식의 string을 합쳐 string으로 반환
        """
        concated_script = "\n".join(scripts)
        return concated_script

    def write_script(self, script: str, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(script)

    def make_empty_file(self, file: str, encoding: str) -> None:
        try:
            file_pointer = open(file=file, mode="w", encoding=encoding)
            file_pointer.close()
        except BaseException:
            print("some file error accured => make_empty_file")
        print(f"create {file} empty file")

    def get_child_dirs(self, path: str, parent_pattern: str, child_pattern: str) -> List[str]:
        """
        데이터가 있는 하위 폴더들을 불러옴
        정규식을 이용하여 최대한 depth에 상관없이 처리 가능하게 함.

        [TODO] : 만약 폴더구조가 음성(wav,pcm), 라벨과 무관하나 비슷한경우, 엉뚱한 파일을 바라보거나 무의미한 중복데이터가 생길 수 있음.
        """
        child_dirs = []
        for parents_dirs in tqdm(Path(path).rglob(parent_pattern)):
            child_dirs.extend(list(Path(parents_dirs).glob(child_pattern)))

        child_dirs = list(map(str, child_dirs))  # convert_string

        return child_dirs

    def preprocess(self):
        pass

    def get_domain_to_paths_dict(self, main_domains: List[str], folder_paths: List[str]) -> DefaultDict[str, List]:
        """
        메인 도메인별 폴더 패스 리스트를 가진 dict을 반환

        ex)
        folder_paths:
        ['/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D20/G02/S000009',
        '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D20/G02/S000010',
        '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000008',
        '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000009',
        '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000010',]

        main_domains:
        ['D20','D21']

        return: {
                D20:['/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D20/G02/S000009',
                    '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D20/G02/S000010',]
                D21:['/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000008'
                    '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000009',
                    '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D21/G02/S000010',]
                }
        """
        main_domains.sort()
        folder_paths.sort()

        domain_to_path_dict = defaultdict(list)

        # two pointer
        folder_paths_idx = 0
        main_domains_idx = 0

        while folder_paths_idx < len(folder_paths) and main_domains_idx < len(main_domains):
            # main_domain = main_domains[main_domains_idx]
            # folder_path = folder_paths[folder_paths_idx]

            while folder_paths_idx < len(folder_paths):
                main_domain = main_domains[main_domains_idx]
                folder_path = folder_paths[folder_paths_idx]
                if main_domain in folder_path:
                    break
                else:
                    folder_paths_idx += 1
                # ex) main_domain이 D22라면 D22 문자열을 가진 폴더의 인덱스까지 idx를 이동

            # 현재 folder_paths_idx: main_domain을 포함하는 문자열의 path의 첫번째 idx
            while folder_paths_idx < len(folder_paths):
                main_domain = main_domains[main_domains_idx]
                folder_path = folder_paths[folder_paths_idx]
                if main_domain in folder_path:
                    # 현재 도메인을 포함하는 path일 경우 append
                    domain_to_path_dict[main_domain].append(folder_path)
                    folder_paths_idx += 1
                else:
                    break

            main_domains_idx += 1

        return domain_to_path_dict

    def train_test_split(
        self,
        total_path: List[str],
        domain_to_paths: Dict[str, List[str]],
        table: Dict[str, int],
    ) -> Union[List[str], List[str]]:
        is_dev = {}
        """
        청소님이 만드신 dev_spliter에서 약간 변형시켜 만든 함수
        """
        # code from 청소
        is_dev = {}

        main_domains = list(domain_to_paths.keys())

        for main_domain in main_domains:
            shuffle(domain_to_paths[main_domain])  # dev set 대상 domain paths를 셔플
            for _ in range(table[main_domain]):
                is_dev[domain_to_paths[main_domain].pop()] = True

        train_path = []
        dev_path = []

        for data_path in total_path:
            if data_path not in is_dev:
                train_path.append(data_path)
            else:
                dev_path.append(data_path)
        # print(train_path)
        # print(dev_path)

        return train_path, dev_path
