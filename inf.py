# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 17:01
@Auth ： Wang Yuyang
@File ：inf.py
@IDE ：PyCharm
"""
import argparse
import os
import pickle

import requests
from tqdm import tqdm

URL = "https://api.infini-gram.io/"


def load_inv_term(n):
    with open(f"n-gram/inv_term_{n}.pkl", "rb") as f:
        inv_term = pickle.load(f)
        return inv_term


def get_inf_gram_count(n_gram_data):
    payload = {
        # "corpus": "v4_piletrain_llama",
        "corpus": "v4_c4train_llama",
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


def save_to_csv(count, n, n_gram):
    if os.path.exists(f"n-gram-count-{n}.csv"):
        with open(f"n-gram-count/n-gram-count-{n}.csv", "a") as f:
            f.write(f"{n_gram},{count}\n")
    else:
        with open(f"n-gram-count/n-gram-count-{n}.csv", "w") as f:
            f.write(f"{n_gram},{count}\n")


def main():
    arg_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('--n', type=int,  help='n-gram')
    args = arg_parser.parse_args()
    n = args.n
    if n is None:
        raise ValueError("Please specify n-gram")
    inv_term = load_inv_term(n)
    print(f'Number of n-grams: {len(inv_term)}')

    for n_gram in tqdm(inv_term):
        count = get_inf_gram_count(n_gram)
        save_to_csv(count, n, n_gram)


if __name__ == '__main__':
    main()
