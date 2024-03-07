# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 17:01
@Auth ： Wang Yuyang
@File ：inf.py
@IDE ：PyCharm
"""
import asyncio
import os

import pandas as pd
from datasets import load_from_disk
from pymongo import MongoClient
from tqdm import tqdm

uri = input("Please input the MongoDB URI: ")

# Create a new client and connect to the server
client = MongoClient(uri)
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)
    exit()

URL = "https://api.infini-gram.io/"
dataset = load_from_disk("snli_with_id")
corpus_name = "v4_dolmasample_olmo"

output_file_name_csv = f"inf-gram/res-{corpus_name}.csv"
output_file_name_parquet = f"inf-gram/res-{corpus_name}.parquet"

import aiohttp  # Make sure to import aiohttp


async def get_inf_gram_count(n_gram_data):
    payload = {
        "corpus": corpus_name,
        "query_type": "count",
        "query": n_gram_data,
    }

    headers = {"Content-Type": "application/json"}

    # Using aiohttp.ClientSession for making asynchronous HTTP requests
    async with aiohttp.ClientSession() as session:
        try:
            # Sending the POST request asynchronously
            async with session.post(URL, json=payload, headers=headers, timeout=10) as response:

                # Parsing the JSON response asynchronously
                json_response = await response.json()

                if "count" not in json_response:
                    print("ERROR!")
                    print(json_response)
                    return -1

                count = json_response["count"]
                return count
        except Exception as e:
            print(e)
            return -1


async def main():
    tqdm_instance = tqdm(dataset)
    data = {}
    if not os.path.exists("inf-gram"):
        os.makedirs("inf-gram")
    for index, example in enumerate(tqdm_instance):
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        doc_id = example["id"]
        label = example["label"]
        if client["infini-gram"][corpus_name].find_one({"doc_id": doc_id}) is not None:
            continue

        premise_count = await get_inf_gram_count(premise)

        hypothesis_count = await get_inf_gram_count(hypothesis)

        tqdm_instance.set_postfix(
            doc_id=doc_id,
            premise_count=premise_count,
            hypothesis_count=hypothesis_count,
        )
        data_entry = {
            'doc_id': doc_id,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "premise_count": premise_count,
            "hypothesis_count": hypothesis_count
        }
        
        data[doc_id] = data_entry
        if client["infini-gram"][corpus_name].find_one({"doc_id": doc_id}) is None:
            client["infini-gram"][corpus_name].insert_one(data_entry)
        else:
            client["infini-gram"][corpus_name].update_one({"doc_id": doc_id}, {"$set": data_entry})

        if index % 100 == 0:
            df = pd.DataFrame.from_dict(data, orient='index')
            df.to_csv(output_file_name_csv)
            df.to_parquet(output_file_name_parquet)

    df = pd.DataFrame.from_dict(data, orient='index')
    df.to_csv(output_file_name_csv)
    df.to_parquet(output_file_name_parquet)


if __name__ == '__main__':
    asyncio.run(main())
