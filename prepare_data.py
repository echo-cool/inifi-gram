# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 12:39
@Auth ： Wang Yuyang
@File ：prepare_data.py
@IDE ：PyCharm
"""
from datasets import load_dataset, concatenate_datasets
import uuid

from datasets import load_dataset, concatenate_datasets


# Function to add an ID based on the example's index
def add_id(example, idx):
    example['id'] = idx  # Use the index as the ID
    return example


def add_id_and_save(dataset, save_path):
    # Adding an ID to each example in the dataset using its index
    dataset_with_id = dataset.map(add_id, with_indices=True)

    # Saving the modified dataset to disk
    dataset_with_id.save_to_disk(save_path)


def main():
    # Load the SNLI dataset
    dataset = load_dataset("snli")

    # Concatenate the train, validation, and test sets
    data = concatenate_datasets([
        dataset["train"],
        dataset["validation"],
        dataset["test"]
    ])

    # Specify the path where you want to save the dataset with IDs
    save_path = 'snli_with_id'

    # Add IDs to the dataset and save it
    add_id_and_save(data, save_path)


if __name__ == "__main__":
    main()
