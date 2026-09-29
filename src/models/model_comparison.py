from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import gc

MODELS_TO_TEST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

TEST_PROMPTS = [
    "Quelle est la capitale de la France ?",
    "Calcule 15 fois 7.",
    "Explique la photosynthèse en deux phrases.",
    "Raconte-moi une blague.",
    "Traduis 'good morning' en chinois.",
]


def load_and_measure(model_name):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Chargement de", model_name, "...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name,torch.float16,"cuda")
    load_time = time.time() - load_start

    latencies = []
    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages,False, True)
        inputs = tokenizer(text,"pt").to(model.device)

        gen_start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        gen_time = time.time() - gen_start
        latencies.append(gen_time)

        response_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("  >", prompt, "->", response_text[:80])

    avg_latency = sum(latencies) / len(latencies)
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "model": model_name,
        "load_time_s": round(load_time, 2),
        "avg_latency_s": round(avg_latency, 2),
        "peak_memory_gb": round(peak_memory_gb, 2),
    }

    del model   
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return result


if __name__ == "__main__":
    results = []
    for model_name in MODELS_TO_TEST:
        results.append(load_and_measure(model_name))

    print("\n--- Tableau comparatif ---")
    print("Modèle | Chargement (s) | Latence moy. (s) | Mémoire GPU (Go)")
    for r in results:
        print(
            r["model"], "|",
            r["load_time_s"], "|",
            r["avg_latency_s"], "|",
            r["peak_memory_gb"]
        )