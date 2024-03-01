# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 21:00
@Auth ： Wang Yuyang
@File ：run_query.py
@IDE ：PyCharm
"""
import pickle

import requests
from tqdm import tqdm

query = pickle.load(open("query/inv_sentence.pkl", "rb"))

URL = "https://api.infini-gram.io/"


def get_inf_gram_count(query):
    payload = {
        "corpus": "v4_rpj_llama_s4",
        "query_type": "count",
        "query": query,
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
    data = {}
    for sentence in tqdm(query):
        count = get_inf_gram_count(sentence)
        data[sentence] = count

    pickle.dump(data, open("query/inv_sentence_count.pkl", "wb"))


if __name__ == '__main__':
    main()
