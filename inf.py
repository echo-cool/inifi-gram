# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 17:01
@Auth ： Wang Yuyang
@File ：inf.py
@IDE ：PyCharm
"""

import pandas as pd
import requests
from datasets import load_from_disk
from tqdm import tqdm

URL = "https://api.infini-gram.io/"
dataset = load_from_disk("snli_with_id")


def get_inf_gram_count(n_gram_data):
    payload = {
        "corpus": "v4_rpj_llama_s4",
        "query_type": "count",
        "query": n_gram_data,
    }

    # Headers to specify that we are sending JSON data
    headers = {"Content-Type": "application/json"}

    # Sending the POST request
    response = requests.post(URL, json=payload, headers=headers)

    # print(response, response.json())

    if "count" not in response.json():
        print("ERROR!")
        print(response.json())

    count = response.json()["count"]
    return count


def main():
    tqdm_instance = tqdm(dataset)
    data = {}
    for index, example in enumerate(tqdm_instance):
        premise = example["premise"].strip()
        hypothesis = example["hypothesis"].strip()
        doc_id = example["id"]
        label = example["label"]

        premise_count = get_inf_gram_count(premise)

        hypothesis_count = get_inf_gram_count(hypothesis)

        tqdm_instance.set_postfix(
            doc_id=doc_id,
            premise_count=premise_count,
            hypothesis_count=hypothesis_count,
        )

        data[doc_id] = {
            'doc_id': doc_id,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "premise_count": premise_count,
            "hypothesis_count": hypothesis_count
        }

        if index % 100 == 0:
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv("inf-gram/res.csv")
            df.to_parquet("inf-gram/res.parquet")

    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_csv("inf-gram/res.csv")
    df.to_parquet("inf-gram/res.parquet")


if __name__ == '__main__':
    main()
