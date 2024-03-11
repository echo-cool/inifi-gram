def load_example(premise, hypothesis, label_bool_str, tokenizer=None, dp=False):
    lst_prompt = []
    lst_prompt.append(("premise", premise))
    lst_prompt.append(("question", "\nquestion:"))
    lst_prompt.append(("hypothesis", " " + hypothesis))
    lst_prompt.append(("domain_premise", " true or false?\nanswer:"))
    lst_prompt.append(("unconditional_hypothesis", " " + label_bool_str + "."))

    if tokenizer is None:
        return lst_prompt, None

    if dp:
        lst_prompt = []
        lst_prompt.append(("domain_premise", " true or false?\nanswer:"))
        lst_prompt.append(("unconditional_hypothesis", " " + label_bool_str + "."))

    dct_idx = {}
    current_idx = 0
    for k, v in lst_prompt:
        part = tokenizer.tokenize(v)
        dct_idx[k] = (current_idx, current_idx + len(part))
        current_idx += len(part)

    return lst_prompt, dct_idx
