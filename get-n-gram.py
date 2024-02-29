# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 12:03
@Auth ： Wang Yuyang
@File ：get-n-gram.py
@IDE ：PyCharm
"""
import collections
import pickle
import re

from datasets import load_from_disk
from tqdm import tqdm

dataset = load_from_disk("snli_with_id")


def get_n_gram(text, n):
    n_gram = set()
    window = collections.deque(maxlen=n)
    for i, word in enumerate(text):
        window.append(word)
        if i < n - 1:
            continue
        n_gram.add(" ".join(window))
    return n_gram


def get_clean_words(text):
    return re.findall(r'\b\w+\b', text)


def process_data_set(dataset, n=2):
    # term_freq = {}
    inv_term = {}
    count = 0
    for example in tqdm(dataset, desc=f"Processing n={n}"):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        doc_id = example["id"]

        text = premise
        n_gram_data = get_n_gram(get_clean_words(text), n)
        for n_gram in n_gram_data:
            if n_gram not in inv_term:
                inv_term[n_gram] = set()
            inv_term[n_gram].add(doc_id)

        text = hypothesis
        n_gram_data = get_n_gram(get_clean_words(text), n)
        for n_gram in n_gram_data:
            if n_gram not in inv_term:
                inv_term[n_gram] = set()
            inv_term[n_gram].add(doc_id)

        # count += 1
        # if count > 100:
        #     break

    pickle.dump(inv_term, open(f"n-gram/inv_term_{n}.pkl", "wb"))
    # for n_gram in inv_term:
    #     inv_term[n_gram] = list(inv_term[n_gram])
    #
    # with open(f"n-gram/inv_term_{n}.json", "w") as f:
    #     json.dump(inv_term, f)


def main():
    # process_data_set(dataset, n=2)

    for i in range(1, 21):
        process_data_set(dataset, n=i)
        # multiprocessing.Process(target=process_data_set, args=(dataset, i)).start()


if __name__ == "__main__":
    main()
