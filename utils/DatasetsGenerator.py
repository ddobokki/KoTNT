import os
from typing import List, Union
import datasets
import traceback
import re

COMMON_TRAIN_SPLIT_NAMES: List[str] = ["train", "eval", "dev"]


class DatsetsForCausalLM(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="bart",
            version="0.0.1",
            description="BartForCausalLM Datasets",
        ),
    ]
    DEFUALT_WRITER_BATCH_SIZE = 1000
    DEFAULT_CONFIG_NAME = "bart"

    def _info(self) -> datasets.DatasetInfo:
        """
        _info
        _summary_
            _description_

        Returns:
            datasets.DatasetInfo: _description_
        """
        if not self.config.name:
            self.config.name = self.DEFAULT_CONFIG_NAME
        features = datasets.Features(
            {
                "feature_text": datasets.Value("string"),
                "label_text": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description="""\
                        어떤 모델을 타겟으로 하는 CausalLM을 위한 Datasets를 구축합니다.
                        """,
            features=features,
            supervised_keys=None,
            homepage="https://aihub.or.kr/",
            license="https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105",
        )

    def _split_generators(self, _: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """
        _split_generators
        _summary_
        huggingface의 datasets.SplitGenerator를 생성하는 함수입니다.

        Examples:
            [
                datasets.SplitGenerator(
                    name: train
                    gen_kwargs: {
                        data_name_1: path_1,
                        data_name_2: path_2,
                        data_name_3: path_3
                        .....
                    }
                )
            ]

        Args:
            _(datasets.DownloadManager): 이 함수에서는 사용하지 않습니다.

        Returns:
            _type_: List[datasets.SplitGenerator] 타입으로 반환합니다.
        """
        train_split_names = COMMON_TRAIN_SPLIT_NAMES
        data_folders = os.listdir(self.config.data_dir)

        output_result = []

        for split_name in train_split_names:
            text_paths = list()
            for data_folder in data_folders:
                data_files = list(
                    filter(
                        lambda x: x.find(split_name) > -1, os.listdir(os.path.join(self.config.data_dir, data_folder))
                    )
                )
                for data_file in data_files:
                    text_paths.append(os.path.join(self.config.data_dir, data_folder, data_file))

            insert_gen_kwargs = {
                "text_paths": text_paths,
            }

            split_generator = datasets.SplitGenerator(
                name=datasets.NamedSplit(split_name), gen_kwargs=insert_gen_kwargs
            )

            output_result.append(split_generator)
        return output_result

    def _generate_examples(
        self,
        text_paths: list[os.PathLike],
    ) -> Union[int, dict]:
        """
        _generate_examples
        _summary_
            Datasets에 들어갈 데이터를 만드는 구간입니다.

        Yields:
            _type_: 0000001,
            {
                "transcription": "안녕하세요 바트",
            }
            와 같은 값들을 반환하게 된다.

        """
        try:
            sent_pair = dict()
            for text_path in text_paths:
                with open(text_path, "r") as text_file:
                    for text_line in text_file:
                        try:
                            sent = text_line.split("::")[1]
                            brackets = re.findall(r"\(.*?\)", sent)  # 괄호 찾음
                            if brackets and len(brackets) % 2 == 0:  # bracket이 있고 짝수개면
                                for i in range(0, len(brackets), 2):
                                    if not bool(re.search(r"\d", brackets[i])):
                                        sent = sent.replace(brackets[i + 1], brackets[i])
                                try:
                                    sent1 = self.sentence_filter(sent, mode="spelling")
                                    if not bool(re.search(r"\d", sent1)):
                                        # 숫자가 없으면
                                        continue
                                    sent2 = self.sentence_filter(sent, mode="phonetic")
                                    sent_pair[sent1] = sent2
                                except Exception:
                                    continue
                        except IndexError:
                            print("처리할 것이 없거나, 음성은 있지만 labels 없는 경우 발생 -> 건너뜀:", text_line)
                            continue
            for idx, (key, value) in enumerate(sent_pair.items()):
                yield idx, {"feature_text": key, "label_text": value}
        except Exception:
            print("\n에러발생 로그를 확인하세요\n" + traceback.format_exc())

    def bracket_filter(self, sentence, mode="phonetic"):
        new_sentence = str()

        if mode == "phonetic":
            flag = False

            for ch in sentence:
                if ch == "(" and flag is False:
                    flag = True
                    continue
                if ch == "(" and flag is True:
                    flag = False
                    continue
                if ch != ")" and flag is False:
                    new_sentence += ch
            if flag:
                raise ValueError("Unsupported mode : {0}".format(sentence))

        elif mode == "spelling":
            flag = True

            for ch in sentence:
                if ch == "(":
                    continue
                if ch == ")":
                    if flag is True:
                        flag = False
                        continue
                    else:
                        flag = True
                        continue
                if ch != ")" and flag is True:
                    new_sentence += ch

        else:
            raise ValueError("Unsupported mode : {0}".format(mode))

        return new_sentence

    def special_filter(self, sentence, mode="phonetic", replace=None):
        SENTENCE_MARK = ["?", "!", "."]
        NOISE = ["o", "n", "u", "b", "l"]
        EXCEPT = ["/", "+", "*", "-", "@", "$", "^", "&", "[", "]", "=", ";", ","] + SENTENCE_MARK

        new_sentence = str()
        for idx, ch in enumerate(sentence):
            if ch not in SENTENCE_MARK:
                if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == "/":
                    continue

            if ch == "#":
                new_sentence += "샾"

            elif ch == "%":
                if mode == "phonetic":
                    new_sentence += replace
                elif mode == "spelling":
                    new_sentence += "%"

            elif ch not in EXCEPT:
                new_sentence += ch

        pattern = re.compile(r"\s\s+")
        new_sentence = re.sub(pattern, " ", new_sentence.strip())
        return new_sentence

    def sentence_filter(self, raw_sentence, mode, replace=None):
        return self.special_filter(self.bracket_filter(raw_sentence, mode), mode, replace)
