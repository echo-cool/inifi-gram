import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_example
import requests
from dotenv import load_dotenv
import os

load_dotenv()
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY")
bool_str_mapping = {
    True: "true",
    False: "false"
}

def get_together_ai(prompt, model, max_tokens, stop=["</s>"]):
    url = "https://api.together.xyz/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stop": stop,
        "logprobs": 1,
        "echo": True
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {TOGETHER_AI_API_KEY}"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=5)
    if response.status_code == 200:
        if response.text:
            res = response.json()
            return res
        else:
            raise Exception("Failed to get response from Together AI: Empty response")
    else:
        raise Exception(f"Failed to get response from Together AI: HTTP {response.status_code}")



def main(model, tokenizer):
    pickle_file_path = f"together-ai/snli_{model.replace('/', '-')}-raw.pkl"
    parquet_file_path = f"together-ai/snli_{model.replace('/', '-')}.parquet"
    existing_dct = pickle.load(open(pickle_file_path, "rb"))

    # get domain conditional logprob
    dct_dp_logprob = {}
    test_premise = ""
    test_hypothesis = ""
    for b_str in ["true", "false"]:
        lst_prompt, dct_idx = load_example(test_premise, test_hypothesis, b_str, tokenizer, dp=True)
        dct_prompt = dict(lst_prompt)
        dp_prompt = dct_prompt["domain_premise"] + dct_prompt["unconditional_hypothesis"]
        res = get_together_ai(dp_prompt, model, 1)
        res = res["prompt"][0]["logprobs"]

        max_token_idx = max(dct_idx.values(), key=lambda x: x[1])[1]
        assert max_token_idx == len(res["tokens"]), "Token length mismatch in domain conditional logprob extraction."

        dct_dp_logprob[b_str] = sum(res["token_logprobs"][dct_idx["unconditional_hypothesis"][0]:dct_idx["unconditional_hypothesis"][1]])

    res_dct = {}
    for k, v in tqdm(existing_dct.items(), desc="Processing dataset", unit=" example"):
        doc_id = k

        timestamp = v["timestamp"]
        premise = v["premise"]
        hypothesis = v["hypothesis"]
        label = v["label"]
        target_bool = v["target_bool"]
        raw_output = v["raw_output"]
        predict_bool = v["predict_bool"]

        res_dct[doc_id] = {
            "timestamp": timestamp,
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "raw_output": raw_output,
            "predict_bool": predict_bool
        }

        for b_str in ["true", "false"]:
            logprobs = v[f"logprob_{b_str}"]["token_logprobs"]
            logprobs[0] = 0  # remove the logprob for the first token

            # sequence logprob
            sequence_logprob = sum(logprobs)
            res_dct[doc_id][f"sequence_logprob_{b_str}"] = sequence_logprob
        sequence_logprob_bool = res_dct[doc_id]["sequence_logprob_true"] > res_dct[doc_id]["sequence_logprob_false"]

        for b_str in ["true", "false"]:
            tokens = v[f"logprob_{b_str}"]["tokens"]
            logprobs = v[f"logprob_{b_str}"]["token_logprobs"]
            logprobs[0] = 0

            # check for token match
            _, dct_idx = load_example(premise, hypothesis, b_str, tokenizer)
            max_token_idx = max(dct_idx.values(), key=lambda x: x[1])[1]
            if max_token_idx != len(tokens):
                print(f"Token length mismatch. Cannot perform logprob extraction.")
                break

            # standard logprob
            lm_logprob = sum(logprobs[dct_idx["unconditional_hypothesis"][0]:dct_idx["unconditional_hypothesis"][1]])
            # average logprob
            avg_logprob = lm_logprob / (dct_idx["unconditional_hypothesis"][1] - dct_idx["unconditional_hypothesis"][0])
            # dcpmi
            dcpmi_logprob = lm_logprob - dct_dp_logprob[b_str]

            # store
            res_dct[doc_id][f"lm_logprob_{b_str}"] = lm_logprob
            res_dct[doc_id][f"avg_logprob_{b_str}"] = avg_logprob
            res_dct[doc_id][f"dcpmi_logprob_{b_str}"] = dcpmi_logprob
        else:
            lm_logprob_bool = res_dct[doc_id]["lm_logprob_true"] > res_dct[doc_id]["lm_logprob_false"]
            res_dct[doc_id]["lm_logprob_bool"] = bool_str_mapping[lm_logprob_bool]
            avg_logprob_bool = res_dct[doc_id]["avg_logprob_true"] > res_dct[doc_id]["avg_logprob_false"]
            res_dct[doc_id]["avg_logprob_bool"] = bool_str_mapping[avg_logprob_bool]
            dcpmi_logprob_bool = res_dct[doc_id]["dcpmi_logprob_true"] > res_dct[doc_id]["dcpmi_logprob_false"]
            res_dct[doc_id]["dcpmi_logprob_bool"] = bool_str_mapping[dcpmi_logprob_bool]
        res_dct[doc_id]["sequence_logprob_bool"] = bool_str_mapping[sequence_logprob_bool]
        res_dct[doc_id]["target_bool"] = target_bool


    df = pd.DataFrame.from_dict(res_dct, orient="index")
    df.to_parquet(parquet_file_path)

if __name__ == "__main__":
    model = "allenai/OLMo-7B"
    # model = "allenai/OLMo-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    main(model, tokenizer)
