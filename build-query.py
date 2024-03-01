# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 20:56
@Auth ： Wang Yuyang
@File ：build-query.py
@IDE ：PyCharm
"""

import pickle

from datasets import load_from_disk
from tqdm import tqdm

dataset = load_from_disk("snli_with_id")

inv_sentence = {}


def main():
    for example in tqdm(dataset, desc="Building query"):
        premise = example["premise"].strip()
        hypothesis = example["hypothesis"].strip()
        doc_id = example["id"]
        label = example["label"]

        if premise not in inv_sentence:
            inv_sentence[premise] = set()
        inv_sentence[premise].add(tuple([doc_id, 0]))

        if hypothesis not in inv_sentence:
            inv_sentence[hypothesis] = set()
        inv_sentence[hypothesis].add(tuple([doc_id, 1]))

    pickle.dump(inv_sentence, open("query/inv_sentence.pkl", "wb"))


if __name__ == '__main__':
    main()
