import pickle


def main(model):
    pickle_file_path = f"together-ai/snli_{model.replace('/', '-')}-raw.pkl"
    existing_dct = pickle.load(open(pickle_file_path, "rb"))

    for k, v in existing_dct.items():
        print(v["logprob_true"])
        print(v["logprob_false"])
        break


if __name__ == "__main__":
    model = "allenai/OLMo-7B"
    main(model)
