# -*- coding: utf-8 -*-
"""
@Time : 3/6/2024 9:03 PM
@Auth : Wang Yuyang
@File : async.py
@IDE  : PyCharm
"""
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import asyncio
import json
import logging
import os
from functools import cache
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import pandas as pd
import yaml
from datasets import load_from_disk
from elasticsearch import Elasticsearch, AsyncElasticsearch
from elasticsearch.exceptions import NotFoundError
from tqdm import tqdm
from pymongo.mongo_client import MongoClient

uri = input("Please input the MongoDB URI: ")
# Create a new client and connect to the server
client = MongoClient(uri)

logger = logging.getLogger(__name__)

PROJECT_ROOT = (Path(__file__).parent / ".." / "..").resolve()
DEFAULT_CONFIG_LOCATION = PROJECT_ROOT / "es_config.yml"
index_name = "docs_v1.5_2023-11-02"
dataset = load_from_disk("../snli_with_id")


@cache
def es_init(config: Path = DEFAULT_CONFIG_LOCATION, timeout: int = 30) -> Elasticsearch:
    """
    :param config: Path to the config yaml file, containing `cloud_id` and `api_key` fields.
    :return: Authenticated ElasticSearch client.
    """
    with open(config) as file_ref:
        config = yaml.safe_load(file_ref)

    cloud_id = config["cloud_id"]
    api_key = config.get("api_key", os.getenv("ES_API_KEY", None))
    if not api_key:
        raise RuntimeError(
            f"Please specify ES_API_KEY environment variable or add api_key to {DEFAULT_CONFIG_LOCATION}."
        )

    es = Elasticsearch(
        cloud_id=cloud_id,
        api_key=api_key,
        retry_on_timeout=True,
        http_compress=True,
        request_timeout=timeout,
    )

    return es


es = es_init()


def get_inf_gram_count(text_data, doc_id, label, type):
    check = client["es_search"][index_name].find_one({"doc_id": doc_id, "type": type})
    if check is not None:
        return None

    query = {
        'bool': {
            'should': [
                {'match_phrase': {'text': text_data}}
            ],
            'minimum_should_match': 1
        }
    }
    res_raw = es.count(
        index=index_name,
        body={'query': query}
    )
    count = res_raw['count']
    res = {
        "doc_id": doc_id,
        "count": count,
        "count_raw": json.dumps(res_raw.raw),
        "label": label,
        "type": type
    }
    return res


def main():
    tqdm_instance = tqdm(dataset, desc="Submit task", unit=" example")
    data = {}
    all_tasks = []
    if not os.path.exists("../inf-gram"):
        os.makedirs("../inf-gram")
    with ThreadPoolExecutor(20) as pool:
        print(f"submitting {len(dataset)} tasks")
        for index, example in enumerate(tqdm_instance):
            premise = example["premise"].strip()
            hypothesis = example["hypothesis"].strip()
            doc_id = example["id"]
            label = example["label"]
            all_tasks.append(pool.submit(get_inf_gram_count, premise, doc_id, label, type="premise"))
            all_tasks.append(pool.submit(get_inf_gram_count, hypothesis, doc_id, label, type="hypothesis"))
        print(f"Total {len(all_tasks)} tasks")
        for future in tqdm(as_completed(all_tasks)):
            result = future.result()
            if result is None:
                continue
            if result["doc_id"] not in data:
                data[result["doc_id"]] = {}
            data[result["doc_id"]][result["type"]] = result

            if client["es_search"][index_name].find_one({"doc_id": result["doc_id"], "type": result["type"]}) is None:
                client["es_search"][index_name].insert_one(result)
            else:
                client["es_search"][index_name].update_one({"doc_id": result["doc_id"], "type": result["type"]},
                                                           {"$set": result})
        with open("inf-gram/inv_sentence_count.pkl", "wb") as f:
            pickle.dump(data, f)


if __name__ == '__main__':
    main()
