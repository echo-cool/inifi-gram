# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 13:49
@Auth ： Wang Yuyang
@File ：test.py
@IDE ：PyCharm
"""
import pickle

for i in range(1, 21):
    with open(f"n-gram/inv_term_{i}.pkl", "rb") as f:
        inv_term = pickle.load(f)
        print(f"n-gram number: {i}, inv_term length: {len(inv_term)}")

