"""
@Time ： 2024/2/29 18:19
@Auth ： Yizhi Hao
@File ：test_local
@IDE ：PyCharm
"""
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialization
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", torch_dtype=torch.float16)
model = model.to('cuda:0')

# Input sentence
prompt = """Premise: A man inspects the uniform of a figure in some East Asian country.
Hypothesis: he man is sleeping
Judgment: entailment"""
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

# Tokenize the input sentence and get input IDs and attention mask
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Forward pass through the model to get logits
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

# Calculate log probabilities using softmax
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

# Shift log_probs and input_ids to align for calculating the probabilities of the input sequence
# We ignore the first token's probability as it has no previous token to condition on
shifted_log_probs = log_probs[..., :-1, :].contiguous()
shifted_input_ids = input_ids[..., 1:].contiguous()

# Gather the log probabilities of the actual next tokens
gathered_log_probs = torch.gather(shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(-1)

# Calculate the total log probability of the input sentence
# (Summing log probabilities here)
total_log_probability = gathered_log_probs.sum()

# Convert log probability to probability
total_probability = torch.exp(total_log_probability)

print(f"Total log probability of the input sentence: {total_log_probability.item()}")
print(f"Total probability of the input sentence: {total_probability.item()}")
