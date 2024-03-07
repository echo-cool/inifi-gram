# -*- coding: utf-8 -*-
"""
@Time ： 3/6/24 19:10
@Auth ： Wang Yuyang
@File ：test.py
@IDE ：PyCharm
"""
import time

from es_search import es_init, count_documents_containing_phrases

es = es_init()
index_name = "docs_v1.5_2023-11-02"


def get_inf_gram_count(n_gram_data):
    start_time = time.time()
    num = count_documents_containing_phrases(
        index_name,
        n_gram_data
    )
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
    return num


if __name__ == '__main__':
    while True:
        n_gram_data = input("Enter n-gram data: ")
        print(get_inf_gram_count(n_gram_data))
