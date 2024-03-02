# -*- coding: utf-8 -*-
"""
@Time : 3/1/2024 10:59 PM
@Auth : Wang Yuyang
@File : aws-prompt.py
@IDE  : PyCharm
"""
import os
from itertools import islice
from datetime import datetime

from datasets import load_from_disk
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import pandas as pd
from transformers import pipeline

dataset = load_from_disk("snli_with_id")
pipe = pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-7B-Base", device=0)
id_label_mapping = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
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
    output = pipe(prompt,max_new_tokens=max_tokens, return_full_text=True, model=model, stop=stop)
    return output[0]['generated_text']


def process_data_set(dataset, model="togethercomputer/RedPajama-INCITE-7B-Base", num_instance=None):
    if not os.path.exists("together-ai"):
        os.makedirs("together-ai")

    parquet_file_path = "together-ai/snli_with_prediction.parquet"

    if os.path.exists(parquet_file_path):
        existing_dct = pd.read_parquet(parquet_file_path).set_index('id').to_dict(orient='index')
        existing_ids = set(existing_dct.keys())
    else:
        existing_dct = {}
        existing_ids = set()

    for example in tqdm(islice(dataset, num_instance), desc="Processing dataset", unit=" example"):
        doc_id = example["id"]

        if doc_id in existing_ids:
            tqdm.write(f"Skipping {doc_id} as it already exists in the dataset.")
            continue

        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label_id = example["label"]

        if label_id not in id_label_mapping:
            tqdm.write(f"Skipping {doc_id} as it does not have a valid label.")
            continue

        label = id_label_mapping[label_id]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # possible leak to the development set as the template contains 1 example
        predict_tmpl = get_jinja_environment().get_template("snli.tpl")
        predict_prompt = predict_tmpl.render(premise=premise, hypothesis=hypothesis)
        raw_prediction = get_together_ai(predict_prompt, model, 50, stop=["</s>", "\n"])

        raw_prediction_id = raw_prediction.strip().lower()
        if raw_prediction_id not in label_id_mapping:
            raw_prediction_id = "invalid"
        clean_prediction_id = label_id_mapping[raw_prediction_id]

        if clean_prediction_id == -1:
            tqdm.write(f"Skipping rationale for {doc_id} as it has an invalid prediction.")
            raw_rationale = ""
        else:
            rationale_tmpl = get_jinja_environment().get_template("snli_rationale.tpl")
            rationale_prompt = rationale_tmpl.render(premise=premise, hypothesis=hypothesis,
                                                     judgment=id_verb_mapping[clean_prediction_id])
            raw_rationale = get_together_ai(rationale_prompt, model, 150, stop=["</s>"])

        existing_dct[doc_id] = {
            "timestamp": timestamp,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "prediction_raw": raw_prediction.strip(),
            "prediction_id": clean_prediction_id,
            "rationale": raw_rationale.strip()
        }

        existing_ids.add(doc_id)
        if doc_id % 100 == 0:
            df = pd.DataFrame.from_dict(existing_dct, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'id'}, inplace=True)
            df.to_parquet(parquet_file_path)

    df = pd.DataFrame.from_dict(existing_dct, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'id'}, inplace=True)
    df.to_parquet(parquet_file_path)


def main():
    process_data_set(dataset, model="togethercomputer/RedPajama-INCITE-7B-Base")


if __name__ == "__main__":
    main()