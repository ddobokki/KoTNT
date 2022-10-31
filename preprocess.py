import csv
import os
import re

from tqdm import tqdm

from utils.sent_filters import sentence_filter
from utils.TRNDataLoader import TRNDataLoader

loader = TRNDataLoader("data/raw")
# t = 0
# for sent in loader.generate():
#     t += 1
sent_pair = {}

j = 0
for sent in tqdm(loader.generate(), total=7326052):
    j += 1

    # 전사를 지우고 숫자가 있으면 전사가 잘못 된 것!
    remove_transcription = re.sub(r"\([^)]*\)", "", sent)
    if bool(re.search(r"\d", remove_transcription)):
        continue

    brackets = re.findall("\(.*?\)", sent)  # 괄호 찾음
    if brackets and len(brackets) % 2 == 0:  # bracket이 있고
        for i in range(0, len(brackets), 2):
            if not bool(re.search(r"\d", brackets[i])):
                sent = sent.replace(brackets[i + 1], brackets[i])
    else:
        continue
    try:
        sent1 = sentence_filter(sent, mode="spelling")
        if not bool(re.search(r"\d", sent1)):
            # 숫자가 없으면
            continue
        sent2 = sentence_filter(sent, mode="phonetic")
        sent_pair[sent1] = sent2
    except:
        pass


sent_pair = list(map(lambda x, y: [x, y], list(sent_pair.keys()), list(sent_pair.values())))
# dictionary를 [[key_1, value_1],[key2, value2]...[key_n, value_n]]

save_foder = "data/csv"

if not os.path.exists(save_foder):
    os.mkdir(save_foder)

with open(os.path.join(save_foder, "out.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(sent_pair)

print(len(sent_pair))

# [TODO] 클린 코드, 하드코딩 제거, 모듈화?
