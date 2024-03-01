# -*- coding: utf-8 -*-
"""
@Time ： 2/29/24 13:49
@Auth ： Wang Yuyang
@File ：test.py
@IDE ：PyCharm
"""
import pickle

data = pickle.load(open("query/inv_sentence.pkl", "rb"))

print(data)
