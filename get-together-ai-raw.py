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
import pickle

from datasets import load_from_disk
from tqdm import tqdm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import load_example

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


def process_example(example, model, tokenizer):
    doc_id = example["id"]

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label_id = example["label"]

    if label_id not in id_label_mapping:
        raise Exception(f"Skipping doc_id {doc_id} as it does not have a valid label. Label ID: {label_id}")

    label_bool = label_bool_mapping[label_id]
    target_bool = bool_str_mapping[label_bool]

    label = id_label_mapping[label_id]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dct_logprobs = {}
    for b in [True, False]:
        b_str = bool_str_mapping[b]
        logprobs_prompt, _ = load_example(premise, hypothesis, b_str, tokenizer)
        res = get_together_ai(logprobs_prompt, model, 1)
        res = res["prompt"][0]["logprobs"]
        dct_logprobs[b_str] = res

    predict_tmpl = get_jinja_environment().get_template("snli_binary_predict.tpl")
    predict_prompt = predict_tmpl.render(premise=premise, hypothesis=hypothesis)
    res = get_together_ai(predict_prompt, model, 50, ["</s>"])
    raw_output = res["choices"][0]["text"]
    predict_bool = re.search(r"^\W*(\w+)", raw_output).group(1).lower()

    if predict_bool == "yes":
        predict_bool = "true"
    if predict_bool == "no":
        predict_bool = "false"
    if predict_bool not in ["true", "false"]:
        predict_bool = "invalid"

    return (doc_id, {
        "timestamp": timestamp,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "target_bool": target_bool,
        "logprob_true": dct_logprobs["true"],
        "logprob_false": dct_logprobs["false"],
        "raw_output": raw_output,
        "predict_bool": predict_bool
    })


def main(max_workers, dataset, model, tokenizer, num_instance):
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pickle_file_path = f"together-ai/snli_{model.replace('/', '-')}-raw.pkl"

        if os.path.exists(pickle_file_path):
            existing_dct = pickle.load(open(pickle_file_path, "rb"))
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

            tasks.append(pool.submit(process_example, example, model, tokenizer))

        for future in tqdm(as_completed(tasks), desc="Processing tasks", unit=" example"):
            if future.exception() is not None:
                print(f"An exception occurred: {future.exception()}")
                continue
            doc_id, dct = future.result()
            existing_ids.add(doc_id)
            existing_dct[doc_id] = dct

            if len(existing_dct) % 100 == 0:
                pickle.dump(existing_dct, open(pickle_file_path, "wb"))

        pickle.dump(existing_dct, open(pickle_file_path, "wb"))


if __name__ == "__main__":
    if not os.path.exists("together-ai"):
        os.makedirs("together-ai")
    max_workers = 5
    dataset = load_from_disk("snli_with_id")
    # model = "allenai/OLMo-7B-Instruct"
    model = "allenai/OLMo-7B"
    # num_instance = 10
    num_instance = None
    tokenizer = None
    main(max_workers, dataset, model, tokenizer, num_instance)
