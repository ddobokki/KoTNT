from asyncio import streams
import os
import random
from typing import DefaultDict, Dict, List, Union
from tqdm import tqdm
from collections import defaultdict
from constants import FileExtension, DataType
from raw_transcripts_abs_base import RawTranscriptsAbsBase


class TranscriptsPreprocess(RawTranscriptsAbsBase):
    def __init__(
        self,
        common_dir: str,
        save_dir: str,
        is_train: bool,
        is_make_dev: bool,
        train_flag_dir_dict: dict,
        parent_pattern: str,
        child_pattern: str,
    ) -> None:
        super().__init__()

        self.common_dir = common_dir
        self.train_flag_dir_dict = train_flag_dir_dict
        self.is_train = is_train
        self.is_make_dev = is_make_dev
        self.parent_pattern = parent_pattern
        self.child_pattern = child_pattern

        if not self.is_train:
            self.is_make_dev = False

        self.save_dir = save_dir
        self.save_names = None
        self.preprocess_data_cnt = 0
        self.unique_preprocess_data_cnt = 0

        self.save_names = self.__get_save_names()

        print(self.is_make_dev)

    def __get_save_names(self) -> List[str]:
        """
        train_type에 따라 저장할 trn 파일의 이름을 가진 리스트 리턴
        "train" -> ["train.trn", "dev.trn"]
        "valid" -> ["eval.trn"]
        """

        save_names = None
        train_file_name = DataType.TRAIN + FileExtension.TRN
        dev_file_name = DataType.DEV + FileExtension.TRN
        eval_file_name = DataType.EVAL + FileExtension.TRN

        if self.is_train:
            save_names = [train_file_name, dev_file_name]
        else:
            save_names = [eval_file_name]

        return save_names

    def __get_domain_to_dirs_dict(self, main_domains: List[str], dirs: List[str]) -> DefaultDict[str, List]:
        """
        메인 도메인별 폴더 패스 리스트를 가진 dict을 반환

        ex)
        dir_paths: [
            '.../D20/G02/S000009',
            '.../D20/G02/S000010',
            '.../D21/G02/S000008',
            '.../D21/G02/S000009',
            '.../D21/G02/S000010',
            ]

        main_domains:
        ['D20','D21']

        return: {
            D20:[
                '.../D20/G02/S000009',
                '.../D20/G02/S000010',
                ]
            D21:[
                '.../D21/G02/S000008'
                '.../D21/G02/S000009',
                '.../D21/G02/S000010',
                ]
            }
        """
        main_domains.sort()
        dirs.sort()

        domain_to_dirs_dict = defaultdict(list)

        # two pointer
        dir_idx = 0
        main_domains_idx = 0

        while dir_idx < len(dirs) and main_domains_idx < len(main_domains):
            while dir_idx < len(dirs):
                main_domain = main_domains[main_domains_idx]
                dir_path = dirs[dir_idx]
                if main_domain in dir_path:
                    break
                else:
                    dir_idx += 1
                # ex) main_domain이 D22라면 D22 문자열을 가진 폴더의 인덱스까지 idx를 이동

            # 현재 dir_idx: main_domain을 포함하는 문자열의 path의 첫번째 idx
            while dir_idx < len(dirs):
                main_domain = main_domains[main_domains_idx]
                dir_path = dirs[dir_idx]
                if main_domain in dir_path:
                    # 현재 도메인을 포함하는 path일 경우 append
                    domain_to_dirs_dict[main_domain].append(dir_path)
                    dir_idx += 1
                else:
                    break

            main_domains_idx += 1

        return domain_to_dirs_dict

    def __convert_script(self, data_dirs: List[str], save_file) -> None:
        """
        data_dir에 있는 txt를 Ksponspeech script로 변환하여 save_file에 저장

        속도가 느린 버그 수정: https://stackoverflow.com/questions/2473783/is-there-a-way-to-circumvent-python-list-append-becoming-progressively-slower
        """
        scripts = []
        data_dirs.sort()
        for data_dir in tqdm(data_dirs, desc="total"):
            # json파일잉 있는 폴더의 txt라는 확장자 파일을 전부 불러오는 부분.
            files = os.listdir(data_dir)

            text_files = []
            files.sort()
            for file in files:
                if FileExtension.TXT in file:
                    text_file = os.path.join(data_dir, file)
                    text_files.append(text_file)
            """
            화자 분리를 할려면 이 부분을 다시 되돌려 놓으세요.
            test = config_rflct(parent_path, child_txt_paths)
            ....?! 아직 이부분 모릅니다.
            """
            for text_file in tqdm(text_files, desc="convert_script", leave=False):
                """
                건내 받은 단일 txt경로를 ksponspeech script형식으로 바꿔주는 작업.
                text_file = '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech/Validation/D20/G02/S000005/xxxx.txt'
                self.common_dir = '/data4/asr_proj/stt/K-wav2vec/data/KconfSpeech'
                """
                try:
                    # 해당 txt파일로부터 scripts를 불러온 다음 (경로) :: (scripts)식으로 만들어 list식으로 만드는 부분.
                    with open(text_file, "r", encoding="utf-8") as f:
                        read_from_file = f.read()
                        if read_from_file.stripe():
                            # label이 없는 경우 건너뜀
                            scripts.append(read_from_file)
                        else:
                            print("label이 없는 데이터는 건너뜁니다.")
                except BaseException:
                    pass
        self.preprocess_data_cnt += len(scripts)  # 확인용
        print("전체 처리된 데이터 cnt:", len(scripts))
        unique_scripts = list(set(scripts))
        self.unique_preprocess_data_cnt += len(unique_scripts)
        print("중복 제거된 데이터 cnt:", len(unique_scripts))
        concated_script = self.concat_scripts(scripts=unique_scripts)
        self.write_script(script=concated_script, path=save_file)

        return None

    def __train_test_split(
        self,
        data_dirs: List[str],
        domain_to_dirs: Dict[str, List[str]],
        domain_devset_cnt: Dict[str, int],
    ) -> Union[List[str], List[str]]:
        """
        공통으로 쓰일 수 있을지?
        dev를 골라내는 domain_devset_cnt 형식이 동일하면 가능할 수도?

        data_dirs : 데이터를 가지고 있는 모든 "폴더"의 경로를 가지는 리스트
        domain_to_dirs : 도메인 별로 묶인 폴더 경로의 dict
        domain_devset_cnt: 도메인당 몇개의 dev 셋을 만들 건지를 저장한 dict

        return train_dirs, dev_dirs, 만약 is_make_dev가 False면 dev_dirs는 빈 리스트를 return

        이곳에서 쓰이는 main_domain -> 바꿀예정
        """
        is_dev = {}

        main_domains = list(domain_to_dirs.keys())

        for main_domain in main_domains:
            random.shuffle(domain_to_dirs[main_domain])  # dev set 대상 domain paths를 셔플
            for _ in range(domain_devset_cnt[main_domain]):
                is_dev[domain_to_dirs[main_domain].pop()] = True

        train_dirs = []
        dev_dirs = []

        for data_dir in data_dirs:
            if data_dir not in is_dev:
                train_dirs.append(data_dir)
            else:
                dev_dirs.append(data_dir)

        return train_dirs, dev_dirs

    def preprocess(self, domain_devset_cnt: Dict[str, int]) -> bool:
        try:
            data_dir = os.path.join(self.common_dir, self.train_flag_dir_dict[self.is_train])
            # print(path)
            child_dirs = self.get_child_dirs(
                path=data_dir,
                parent_pattern=self.parent_pattern,
                child_pattern=self.child_pattern,
            )
            child_dirs.sort()

            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)
            if self.is_make_dev:
                assert domain_devset_cnt, "domain_devset_cnt not found"

                train_file = os.path.join(self.save_dir, self.save_names[0])
                dev_file = os.path.join(self.save_dir, self.save_names[1])

                self.make_empty_file(file=train_file, encoding="utf-8")
                self.make_empty_file(file=dev_file, encoding="utf-8")

                main_domains = list(domain_devset_cnt.keys())
                domain_to_dirs_dict = self.__get_domain_to_dirs_dict(main_domains=main_domains, dirs=child_dirs)

                train_dirs, dev_dirs = self.__train_test_split(
                    data_dirs=child_dirs,
                    domain_to_dirs=domain_to_dirs_dict,
                    domain_devset_cnt=domain_devset_cnt,
                )

                print(data_dir, "train, dev 시작합니다.")
                self.__convert_script(data_dirs=train_dirs, save_file=train_file)
                self.__convert_script(data_dirs=dev_dirs, save_file=dev_file)
            else:

                train_eval_file = os.path.join(self.save_dir, self.save_names[0])
                self.make_empty_file(file=train_eval_file, encoding="utf-8")

                print(data_dir, "eval 시작합니다.")
                self.__convert_script(data_dirs=child_dirs, save_file=train_eval_file)
        except Exception:
            return False
        return True
