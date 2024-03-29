"""
@Time ： 2024/2/29 16:10
@Auth ： Yizhi Hao
@File ：get-together-ai.py
@IDE ：PyCharm
"""

import os
from itertools import islice
import requests
from datetime import datetime

from datasets import load_from_disk
from tqdm import tqdm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")

id_label_mapping = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

label_bool_mapping = {
    0: True,
    1: False,
    2: False
}

bool_str_mapping = {
    True: "true",
    False: "false"
}

label_id_mapping = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "invalid": -1
}

id_verb_mapping = {
    0: "entails",
    1: "is neutral to",
    2: "contradicts",
}


def get_jinja_environment() -> Environment:
    return Environment(loader=FileSystemLoader("templates"))


def get_together_ai(prompt, model, max_tokens, stop=["</s>"]):
    url = "https://api.together.xyz/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
        "logprobs": 1,
        "echo": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=5)
    if response.status_code == 200:
        if response.text:
            res = response.json()
            return res
        else:
            raise Exception("Failed to get response from Together AI: Empty response")
    else:
        raise Exception(f"Failed to get response from Together AI: HTTP {response.status_code}")


def process_example(example, model):
    doc_id = example["id"]

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label_id = example["label"]

    if label_id not in id_label_mapping:
        raise Exception(f"Skipping doc_id {doc_id} as it does not have a valid label. Label ID: {label_id}")

    label_bool = label_bool_mapping[label_id]
    label_bool_str = bool_str_mapping[label_bool]

    label = id_label_mapping[label_id]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logprobs_tmpl = get_jinja_environment().get_template("snli_binary_logprobs.tpl")

    dct_logprobs = {}
    for b in [True, False]:
        b_str = bool_str_mapping[b]
        logprobs_prompt = logprobs_tmpl.render(premise=premise, hypothesis=hypothesis, bool_str=b_str)
        res = get_together_ai(logprobs_prompt, model, 1)
        logprob = sum(res["prompt"][0]["logprobs"]["token_logprobs"][1:])
        dct_logprobs[b_str] = logprob

    predict_bool_logprob = dct_logprobs["true"] > dct_logprobs["false"]
    label_bool_logprob = bool_str_mapping[predict_bool_logprob]

    predict_tmpl = get_jinja_environment().get_template("snli_binary_predict.tpl")
    predict_prompt = predict_tmpl.render(premise=premise, hypothesis=hypothesis)
    res = get_together_ai(predict_prompt, model, 50, ["</s>"])
    raw_output = res["choices"][0]["text"]
    label_predict = re.search(r"^\W*(\w+)", raw_output).group(1).lower()

    if label_predict == "yes":
        label_predict = "true"
    if label_predict == "no":
        label_predict = "false"
    if label_predict not in ["true", "false"]:
        label_predict = "invalid"

    return (doc_id, {
        "timestamp": timestamp,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "label_bool_str": label_bool_str,
        "label_true_logprob": dct_logprobs["true"],
        "label_false_logprob": dct_logprobs["false"],
        "label_bool_logprob": label_bool_logprob,
        "raw_output": raw_output,
        "label_bool_predict": label_predict
    })


def main(max_workers, dataset, model, num_instance):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        parquet_file_path = f"together-ai/snli_{model.replace('/', '-')}.parquet"

        if os.path.exists(parquet_file_path):
            existing_dct = pd.read_parquet(parquet_file_path).set_index('id').to_dict(orient='index')
            existing_ids = set(existing_dct.keys())
        else:
            existing_dct = {}
            existing_ids = set()

        tasks = []
        for example in tqdm(islice(dataset, num_instance), desc="Submitting tasks", unit=" example"):
            doc_id = example["id"]

            if doc_id in existing_ids:
                print(f"Skipping {doc_id} as it already exists in the dataset.")
                continue

            tasks.append(pool.submit(process_example, example, model))

        for future in tqdm(as_completed(tasks), desc="Processing tasks", unit=" example"):
            if future.exception() is not None:
                print(f"An exception occurred: {future.exception()}")
                continue
            doc_id, dct = future.result()
            existing_ids.add(doc_id)
            existing_dct[doc_id] = dct

            if len(existing_dct) % 100 == 0:
                df = pd.DataFrame.from_dict(existing_dct, orient='index')
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'id'}, inplace=True)
                df.to_parquet(parquet_file_path)

        df = pd.DataFrame.from_dict(existing_dct, orient='index')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)
        df.to_parquet(parquet_file_path)


if __name__ == "__main__":
    if not os.path.exists("together-ai"):
        os.makedirs("together-ai")
    max_workers = 5
    dataset = load_from_disk("snli_with_id")
    # model = "allenai/OLMo-7B-Instruct"
    model = "allenai/OLMo-7B"
    # num_instance = 10
    num_instance = None
    main(max_workers, dataset, model, num_instance)
