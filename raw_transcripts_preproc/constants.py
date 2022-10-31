#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Preprocess 부분의 상수 정의

* dataclass를 이용한 상수정의 (frozen: 수정 불가)
* class단위로 작성하며, inner class 적극 활용 (계층적 표현으로 가독성을 올리기 위함)

[TODO]:
* BART: 추가할 내용 있으면 계속 추가 예정...

"""
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DownDataInfo:
    """Download된 회사별 데이터를 일관화 체크하기 위한 모듈의 상수 \n
    Download된 파일을 설명할 수 있는 고정 상수들이 사용됨

    Attributes:
        CONFIG_FILE (str): config.yaml \n
        TMAX (str): 티맥스소프트 \n
        ETRI (str): 한국전자통신연구원 \n
        SOLU (str): 솔루게이트
    """

    CONFIG_FILE: str = "config.yaml"
    """ :value: config.yaml """
    TMAX: str = "티맥스소프트"
    """ :value: 티맥스소프트 """
    ETRI: str = "한국전자통신연구원"
    """ :value: 한국전자통신연구원 """
    SOLU: str = "솔루게이트"
    """ :value: 솔루게이트 """


@dataclass
class FileExtension:
    """
    constant.py에 추가 예정 -> 현재 저장 문제있음
    """

    TXT: str = ".txt"
    WAV: str = ".wav"
    TRN: str = ".trn"
    PCM: str = ".pcm"


@dataclass
class DataType:
    """
    constant.py에 추가 예정 -> 현재 저장 문제있음
    """

    TRAIN: str = "train"
    EVAL: str = "eval"
    DEV: str = "dev"
