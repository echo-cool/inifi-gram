import os
import pickle
import aiohttp
import asyncio
from tqdm import tqdm  # Import tqdm for progress bars

URL = "https://api.infini-gram.io/"


async def load_inv_term(n):
    with open(f"n-gram/inv_term_{n}.pkl", "rb") as f:
        inv_term = pickle.load(f)
        return inv_term


async def get_inf_gram_count(session, n_gram_data, semaphore, progress_bar):
    async with semaphore:  # Acquire a semaphore
        payload = {
            "corpus": "v4_c4train_llama",
            "query_type": "count",
            "query": n_gram_data,
        }

        headers = {"Content-Type": "application/json"}

        async with session.post(URL, json=payload, headers=headers) as response:
            resp_json = await response.json()

            if "count" not in resp_json:
                print("ERROR!")
                print(resp_json)
                return n_gram_data, 0

            count = resp_json["count"]
            progress_bar.update(1)  # Update the progress bar here
            return n_gram_data, count


async def save_to_csv(n_gram_counts, n):
    filename = f"n-gram-count/n-gram-count-{n}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        for n_gram, count in n_gram_counts.items():
            f.write(f"{n_gram},{count}\n")


async def main():
    n = 1
    inv_term = await load_inv_term(n)
    print(f'Number of n-grams: {len(inv_term)}')

    n_gram_counts = {}
    semaphore = asyncio.Semaphore(10)  # Limit concurrency to 100

    async with aiohttp.ClientSession() as session:
        # Initialize tqdm progress bar
        with tqdm(total=len(inv_term)) as progress_bar:
            tasks = [get_inf_gram_count(session, n_gram, semaphore, progress_bar) for n_gram in inv_term]
            results = await asyncio.gather(*tasks)

            # Populate the dictionary with results
            for n_gram, count in results:
                n_gram_counts[n_gram] = count

    # Save the collected counts to a CSV file
    await save_to_csv(n_gram_counts, n)


if __name__ == '__main__':
    asyncio.run(main())
