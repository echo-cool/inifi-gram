def load_example(premise, hypothesis, label_bool_str, tokenizer=None):
    lst_prompt = []
    lst_prompt.append(("premise", premise))
    lst_prompt.append(("question", "\nquestion:"))
    lst_prompt.append(("hypothesis", " " + hypothesis))
    lst_prompt.append(("domain_premise", "true or false?\nanswer:"))
    lst_prompt.append(("conditiona_hypothesis", " " + label_bool_str + "."))

    prompt = "".join([v for k, v in lst_prompt])

    if tokenizer is None:
        return prompt, None

    dct_idx = {}
    current_idx = 0
    for k, v in lst_prompt:
        part = tokenizer.tokenize(v)
        dct_idx[k] = (current_idx, current_idx + len(part))
        current_idx += len(part)

    return prompt, dct_idx
