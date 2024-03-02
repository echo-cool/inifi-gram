# -*- coding: utf-8 -*-
"""
@Time : 3/1/2024 12:04 AM
@Auth : Wang Yuyang
@File : aws.py
@IDE  : PyCharm
"""
# SageMaker JumpStart provides APIs as part of SageMaker SDK that allow you to deploy and fine-tune models in network isolation using scripts that SageMaker maintains.

from sagemaker.jumpstart.model import JumpStartModel


model_id = "huggingface-textgeneration1-redpajama-incite-base-7B-v1-fp16"
endpoint_input = {'inputs': 'Once upon a time,'}

model = JumpStartModel(model_id=model_id)
predictor = model.deploy()
response = predictor.predict(endpoint_input)
print(f"Inference:\nInput: {endpoint_input}\nResponse: {response}\n")