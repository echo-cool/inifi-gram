# -*- coding: utf-8 -*-
"""
@Time : 3/3/2024 3:12 PM
@Auth : Wang Yuyang
@File : test.py
@IDE  : PyCharm
"""
from datasets import load_dataset

dataset = load_dataset("togethercomputer/RedPajama-Data-1T", 'default')
print(dataset)