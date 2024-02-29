# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 13:49
@Auth ： Wang Yuyang
@File ：test.py
@IDE ：PyCharm
"""
import pickle

with open("n-gram/inv_term_5.pkl", "rb") as f:
    inv_term = pickle.load(f)
    print(inv_term)