# -*- coding: utf-8 -*-
"""
@Time : 2/29/2024 11:56 PM
@Auth : Wang Yuyang
@File : prompt.py
@IDE  : PyCharm
"""
# Load model directly
from transformers import pipeline

pipe = pipeline("text-generation", model="togethercomputer/RedPajama-INCITE-7B-Base", device=0)

# Prompt the model
prompt = "Hello"
output = pipe(prompt, max_length=50, do_sample=True, temperature=0.7)
print(output)
# Output: import pandas as pd
