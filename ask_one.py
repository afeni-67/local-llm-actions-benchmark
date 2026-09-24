#!/usr/bin/env python3
"""One-shot local Q&A to prove the model responds."""
import os, time
from model_runner import download_model, load_llm

QUESTION = os.environ.get(
    "ASK_QUESTION",
    "In one short sentence: what is the capital of France?",
)

def main():
    path = download_model()
    llm = load_llm(path)
    print("QUESTION:", QUESTION)
    t0 = time.perf_counter()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "Answer briefly and clearly in plain English."},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.2,
        max_tokens=80,
    )
    ans = out["choices"][0]["message"]["content"]
    print("ANSWER:", ans)
    print(f"inference_seconds={time.perf_counter()-t0:.2f}")

if __name__ == "__main__":
    main()
