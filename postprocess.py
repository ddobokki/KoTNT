import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import difflib
import re


def tot_eda(data):
    print("평균 : ", np.mean(data))
    print("중앙값 : ", np.median(data))
    print("최빈값 : ", np.bincount(data).argmax())
    print("최대값 : ", np.max(data))
    print("최소값 : ", np.min(data))
    print("표본 표준편차(자유도) : ", np.std(data, ddof=True))
    print("표본 표준편차(데이터 개수) : ", np.std(data, ddof=False))
    print("표본 분산(자유도) : ", np.var(data, ddof=True))
    print("표본 분산(데이터 개수) : ", np.var(data, ddof=False))
    print(
        "IQR : ", np.subtract(*np.quantile(data, [0.75, 0.25]))
    )  # 또는 np.quantile(data, 0.75)-np.quantile(data, 0.25)
    print("범위 : ", np.max(data) - np.min(data))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        metavar="DIR",
        default="./data/csv/train.csv",
        required=True,
        help="where is your csv data dir?",
    )
    return parser


def main(args) -> None:
    np_eda_ko = np.array([], dtype=int)
    np_eda_num = np.array([], dtype=int)
    np_post_ko = np.array([], dtype=int)
    np_post_num = np.array([], dtype=int)
    csv_path = args.csv_path
    file_dir, file_name = os.path.split(csv_path)
    post_path = os.path.join(file_dir, "post_" + file_name)
    f = open(csv_path, "r", encoding="utf-8")
    post_f = open(post_path, "w", newline="\n")
    wr = csv.writer(post_f)
    rdr = csv.reader(f)
    for idx, line in tqdm(enumerate(rdr)):
        if len(line) >= 3:
            print(f"\n\n{idx} 에서 전처리 feature, label 넘는 데이터 발견")
            print(line)
            continue
        elif len(line) < 2:
            print(f"\n\n{idx} 에서 feature 혹은 label만 있음")
            print(line)
            continue

        len_num_data = len(line[0])
        np_eda_num = np.append(np_eda_num, len_num_data)
        len_ko_data = len(line[1])
        np_eda_ko = np.append(np_eda_ko, len_ko_data)

        # 문자의 유사도를 판단한다.
        answer_bytes = bytes(line[0], "utf-8")
        input_bytes = bytes(line[1], "utf-8")
        answer_bytes_list = list(answer_bytes)
        input_bytes_list = list(input_bytes)

        sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
        similar = sm.ratio()

        if (
            (len_ko_data <= 10 and similar <= 0.1)
            or ((10 < len_ko_data <= 20) and similar <= 0.2)
            or ((20 < len_ko_data <= 50) and similar <= 0.4)
            or (len_ko_data > 50 and similar <= 0.5)
        ):
            chk_last_chr = line[0][-1]
            # 문장의 끝이 동일하지 않은데,
            # 정답 문장이 50글자가 넘고, 마지막글자가 한글이면, 분명 뭔가 짤렸을 가능성이 높다 (15% 데이터 직접 확인)
            if line[0][-1] != line[1][-1]:
                if len_ko_data > 50 or re.search(r"[가-힣]", chk_last_chr):
                    continue
        np_post_num = np.append(np_eda_num, len(line[0]))
        np_post_ko = np.append(np_eda_ko, len(line[1]))
        wr.writerow(line)
    post_f.close()
    f.close()
    print("=== 글자 데이터 통계 ===")
    tot_eda(np_eda_ko)
    print("=== 숫자 데이터 통계 ===")
    tot_eda(np_eda_num)
    print("=== 후처리 글자 데이터 통계 ===")
    tot_eda(np_post_ko)
    print("=== 후처리 숫자 데이터 통계 ===")
    tot_eda(np_post_num)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    def _print_config(config):
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(args)
    main(args)
