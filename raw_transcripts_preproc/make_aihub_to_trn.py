import os
import argparse
import yaml
import constants
from typing import Dict
from setproctitle import setproctitle
from transcripts_preprocess import TranscriptsPreprocess

down_data_info_const = constants.DownDataInfo()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root_dir",
        type=str,
        metavar="DIR",
        default="../data",
        required=True,
        help="where is your data dir?",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        metavar="DIR",
        default="../data/raw",
        required=True,
        help="where is your output scripts dir?",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        required=True,
        help="what is your scenario? ex) 'KconfSpeech'",
    )
    parser.add_argument(
        "--is_train",
        type=int,
        default=1,
        choices=[0, 1],
        required=True,
        help="train is 1 / valid is 0",
    )
    parser.add_argument(
        "--is_make_dev",
        type=int,
        default=0,
        choices=[0, 1],
        required=True,
        help="if you want to make dev dataset, write '1'",
    )
    return parser


def main(args):
    """First Task ===> Data Download 이후 수행
    trn 파일 (wav or pcm 경로 :: script 내용)을 구성합니다.
    ETRI 전사규칙을 따른다 하더라도, 업체별로 폴더 구조가 다르고,
    ETRI 전사규칙을 선택하기위한 (한글, 숫자 등) 공통 모듈을 구동하기 위해서는 trn파일이 필요합니다.

    해당 trn 파일을 생성하기 위한 첫번째 전처리를 시작합니다.

    Process Flow:
        1. argument에서 scenario명 선택
        2. train or valid 중 argument에서 정의한 Task를 처리함
        3. 해당 scenario의 config.yaml을 불러옴 (필수)
        4. TMAX와 SOLU의 경우, 동일한 처리방법을 사용합니다.
            KsponPreprocess의 preprocess로 처리됨.
            업체가 추가될 경우, 해당 부분의 수정이 발생할 수 있음

    Attributes:
        1. down_data_configs : config.yaml에서 불러온 config 값
        2. parent_pattern : txt set을 불러오기 위한 폴더구조 부모 정규식 (추상 클래스 참고)
        3. child_pattern : txt set을 불러오기 위한 폴더구조 자식 정규식 (추상 클래스 참고)
            - 폴더 depth가 다르더라도, 유연하게 최종 폴더를 뽑아낼 수 있도록 정규식 처리
    """
    config_chk_error_list: list[str] = list()
    scenario_dir = os.path.join(args.data_root_dir, args.scenario_name)
    if os.path.isdir(scenario_dir):
        if not os.path.isfile(os.path.join(scenario_dir, down_data_info_const.CONFIG_FILE)):  # config 파일은 필수입니다.
            print("config.yaml 없음 : " + args.scenario_name)
        else:
            down_data_configs = None
            scripts_save_dir = os.path.join(args.output_dir, args.scenario_name + "_scripts")
            with open(os.path.join(scenario_dir, down_data_info_const.CONFIG_FILE)) as config_file:
                down_data_configs = yaml.load(config_file, Loader=yaml.FullLoader)

            build_company = down_data_configs["common"]["구축기관"]
            domain_devset_cnt = down_data_configs["dataset"]["domain_devset_cnt"]
            train_flag_dir_dict = down_data_configs["dir"]["train_flag_dir_dict"]
            parent_pattern = down_data_configs["dir"]["parent_pattern"]
            child_pattern = down_data_configs["dir"]["child_pattern"]

            if build_company == down_data_info_const.TMAX or build_company == down_data_info_const.SOLU:

                is_success = TranscriptsPreprocess(
                    common_dir=scenario_dir,
                    save_dir=scripts_save_dir,
                    is_train=args.is_train,
                    is_make_dev=args.is_make_dev,
                    train_flag_dir_dict=train_flag_dir_dict,
                    parent_pattern=parent_pattern,
                    child_pattern=child_pattern,
                ).preprocess(domain_devset_cnt)

                if not is_success:
                    print("preprocess간 오류 발생 : " + build_company)
            elif build_company is down_data_info_const.ETRI:
                pass
            else:
                print("preprocess.constants에 선언되지 않은 업체는 처리 불가합니다. : " + build_company)
    if config_chk_error_list:
        print("config check 에 실패한 파일 혹은 사유 :: ", config_chk_error_list)


if __name__ == "__main__":
    setproctitle("check_kspon_style")
    parser = get_parser()
    args = parser.parse_args()

    # 이 부분 check!
    def _print_config(config):
        import pprint

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    _print_config(args)
    main(args)
