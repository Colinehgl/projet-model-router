from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import time
import gc
import logging

logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

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
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
    load_time = time.time() - load_start

    latencies = []
    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
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



def load_and_measure_quantized(model_name, bits=4):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Chargement de", model_name, "en", bits, "bit ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="cuda"
    )
    load_time = time.time() - load_start

    latencies = []
    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        gen_time = time.time() - gen_start
        latencies.append(gen_time)

        response_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print("  >", prompt, "->", response_text[:80])

    avg_latency = sum(latencies) / len(latencies)
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    result = {
        "model": model_name + " (" + str(bits) + "-bit)",
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
        r = load_and_measure(model_name)
        results.append(r)
        print("RESULTAT :", r)

    r8 = load_and_measure_quantized("Qwen/Qwen2.5-3B-Instruct", bits=8)
    results.append(r8)
    print("RESULTAT :", r8)

    r4 = load_and_measure_quantized("Qwen/Qwen2.5-3B-Instruct", bits=4)
    results.append(r4)
    print("RESULTAT :", r4)

    print("\n--- Tableau comparatif ---")
    print("Modèle | Chargement (s) | Latence moy. (s) | Mémoire GPU (Go)")
    for r in results:
        print(r["model"], "|", r["load_time_s"], "|", r["avg_latency_s"], "|", r["peak_memory_gb"])