def bart_preprocess(batch, tokenizer):
    model_inputs = tokenizer(
        batch["num_sent"],
        return_length=True,
    )
    labels = tokenizer(
        batch["ko_sent"],
    )
    reversed_labels = list(reversed(labels["input_ids"]))
    reversed_labels.append(tokenizer.eos_token_id)
    model_inputs["labels"] = reversed_labels
    return model_inputs


def t5_preprocess(raw, prompt, tokenizer):
    """"""
    # prompt = "translation_num_to_text"
    train_input = f"""{prompt}: {raw["num_col"]}"""
    label_input = raw["sen_col"]

    # [NOTE]: train이라는 이름은 나중에 바꾸는 게 좋을 듯 valid, test도 있어서 맞지가 않는다.
    train_encoded = tokenizer(train_input, return_attention_mask=False, max_length=240)
    label_encoded = tokenizer(label_input, return_attention_mask=False, max_length=240)

    result = {"sen_col": train_encoded["input_ids"], "num_col": label_encoded["input_ids"]}
    return result


def gpt_preprocess(raw, tokenizer, train_type):
    num_col_text = raw["num_col"]
    sen_col_text = raw["sen_col"]
    BOS = tokenizer.bos_token
    EOS = tokenizer.eos_token

    if train_type == "NTT":
        text = BOS + num_col_text + EOS + BOS + sen_col_text + EOS
    elif train_type == "TTN":
        text = BOS + sen_col_text + EOS + BOS + num_col_text + EOS
    else:
        raise Exception

    raw = tokenizer(text)
    raw["labels"] = raw["input_ids"]

    return raw
