import re


def bracket_filter(sentence, mode="phonetic"):
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


def special_filter(sentence, mode="phonetic", replace=None):
    NOISE = ["o", "n", "u", "b", "l"]
    EXCEPT = ["/", "+", "*", "@", "$", "^", "&", "[", "]", ";", ","]

    new_sentence = str()
    for idx, ch in enumerate(sentence):
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


def sentence_filter(raw_sentence, mode, replace=None):
    return special_filter(bracket_filter(raw_sentence, mode), mode, replace)
