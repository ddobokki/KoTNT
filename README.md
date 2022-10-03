# KoGPT_num_converter

한국어 문장에서 숫자가 들어간 문장을 한글로 변환 해주는 모델입니다.
그 반대도 가능합니다.

숫자를 경우에 따라 읽는 방법이 달라지게 되므로 이를 딥러닝 모델을 활용해 해결하려는 토이 프로젝트입니다.

ex)
- 경제 1번지 <-> 경제 일번지
- 오늘 1시에 보는게 어때? <-> 오늘 한 시에 보는게 어때?

# 사용 데이터
- 본 repo에서는 음성 데이터의 정답 스크립트를 [KoSpeech](https://github.com/sooftware/kospeech) 라이브러리를 사용하여 trn 파일로 전처리하여 사용했습니다.
[KSponSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123)는 이곳에서 다운 받을 수 있습니다.
- 공개할 모델은 이외에도 [KoreanSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=130), [KconfSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=132), [KrespSpeech](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=87)의 스크립트를 학습한 모델입니다.