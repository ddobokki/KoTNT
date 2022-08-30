import re

from utils.sent_filters import sentence_filter
from utils.TRNDataLoader import TRNDataLoader

loader = TRNDataLoader("data/raw")
sent_pair = []
for sent in loader.generate():
    brackets = re.findall("\(.*?\)", sent)  # 괄호 찾음
    if brackets and len(brackets) % 2 == 0:  # bracket이 있고
        for i in range(0, len(brackets), 2):
            if not bool(re.search(r"\d", brackets[i])):
                sent = sent.replace(brackets[i + 1], brackets[i])
    try:
        sent1 = sentence_filter(sent, mode="spelling")
        if not bool(re.search(r"\d", sent1)):
            # 숫자가 없으면
            continue
        sent2 = sentence_filter(sent, mode="phonetic")
        sent_pair.append([sent1, sent2])
    except:
        pass

# [TODO]
# 1. 전사 규칙 잘 못된 애들 필터링 ex 1주일이 (1주일) / (일주일)로 안 되어 있는 문장
# 2. 알고리즘 효율화(현재 상태 - 느림)

print(len(sent_pair))
