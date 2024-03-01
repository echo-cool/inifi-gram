"""
@Time ： 2024/2/29 16:10
@Auth ： Yizhi Hao
@File ：get-together-ai.py
@IDE ：PyCharm
"""

import os
from itertools import islice
import requests

from datasets import load_from_disk
from tqdm import tqdm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
import pandas as pd

load_dotenv()
TOGERHER_AI_API_KEY = os.getenv("TOGERHER_AI_API_KEY")
dataset = load_from_disk("snli_with_id")

id_label_mapping = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

label_id_mapping = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

id_verb_mapping = {
    0: "entails",
    1: "is neutral to",
    2: "contradicts"
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
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TOGERHER_AI_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers)

    return response.json()['choices'][0]['text']


def process_data_set(dataset, model="togethercomputer/RedPajama-INCITE-7B-Base", num_instance=None):
    parquet_file_path = "data/snli_with_prediction.parquet"

    if os.path.exists(parquet_file_path):
        existing_df = pd.read_parquet(parquet_file_path)
        existing_ids = set(existing_df['id'])
    else:
        existing_ids = set()

    for example in tqdm(islice(dataset, num_instance)):
        doc_id = example["id"]

        if doc_id in existing_ids:
            print(f"Skipping {doc_id} as it already exists in the dataset.")
            continue

        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label_id = example["label"]
        label = id_label_mapping[label_id]

        print(f"Processing {doc_id}: {premise} | {hypothesis} | {label}")

        # possible leak to the development set as the template contains 1 example
        predict_tmpl = get_jinja_environment().get_template("snli.tpl")
        predict_prompt = predict_tmpl.render(
            premise=premise,
            hypothesis=hypothesis,
        )
        raw_prediction = get_together_ai(predict_prompt, model, 50, stop=["</s>", "\n"])
        clean_prediction_id = label_id_mapping[raw_prediction.strip().lower()]

        rationale_tmpl = get_jinja_environment().get_template("snli_rationale.tpl")
        rationale_prompt = rationale_tmpl.render(
            premise=premise,
            hypothesis=hypothesis,
            judgment=id_verb_mapping[clean_prediction_id],
        )

        raw_rationale = get_together_ai(rationale_prompt, model, 150, stop=["</s>"])

        df = pd.DataFrame({
            "id": [doc_id],
            "premise": [premise],
            "hypothesis": [hypothesis],
            "label": [label],
            "prediction_raw": [raw_prediction.strip()],
            "prediction_id": [clean_prediction_id],
            "rationale": [raw_rationale.strip()]
        })

        if 'existing_df' in locals():
            existing_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            existing_df = df.copy()

        existing_ids.add(doc_id)

    if 'existing_df' in locals():
        existing_df.to_parquet(parquet_file_path, index=False)


def main():
    process_data_set(dataset, model="togethercomputer/RedPajama-INCITE-7B-Base", num_instance=10)


if __name__ == "__main__":
    main()
