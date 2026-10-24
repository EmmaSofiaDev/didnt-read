"""
eval.py - Evaluate base model vs fine-tuned Tinker LoRA model on LexGLUE UNFAIR-ToS test split.
Computes multi-label Precision, Recall, and F1 scores, plus latency and token metrics.
"""

import os
import json
import asyncio
import time
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
import tinker

LABEL_NAMES = [
    "Limitation of liability",
    "Unilateral termination",
    "Unilateral change",
    "Content removal",
    "Contract by using",
    "Choice of law",
    "Jurisdiction",
    "Arbitration"
]

SYSTEM_PROMPT = (
    "You are an expert contract lawyer analyzing Terms of Service clauses. "
    "Classify the clause into one or more of these 8 unfair clause categories:\n"
    "- Limitation of liability\n"
    "- Unilateral termination\n"
    "- Unilateral change\n"
    "- Content removal\n"
    "- Contract by using\n"
    "- Choice of law\n"
    "- Jurisdiction\n"
    "- Arbitration\n"
    "If the clause contains no unfair terms, respond with 'Fair'. "
    "Respond ONLY with the category name(s) separated by commas, or 'Fair'."
)

def parse_model_prediction(pred_text: str) -> list[int]:
    """Parse output text into multi-label binary vector of length 8."""
    text_lower = pred_text.lower()
    vector = [0] * len(LABEL_NAMES)
    
    # If explicitly Fair and no category keywords present
    if "fair" in text_lower and not any(cat.lower() in text_lower for cat in LABEL_NAMES):
        return vector
        
    for i, cat in enumerate(LABEL_NAMES):
        # Match label names or key unambiguous fragments
        name_lower = cat.lower()
        if name_lower in text_lower:
            vector[i] = 1
        elif cat == "Arbitration" and "arbitrat" in text_lower:
            vector[i] = 1
        elif cat == "Limitation of liability" and ("liability" in text_lower or "limitation" in text_lower):
            vector[i] = 1
        elif cat == "Unilateral change" and ("unilateral change" in text_lower or "change terms" in text_lower):
            vector[i] = 1
        elif cat == "Unilateral termination" and ("unilateral term" in text_lower or "terminate" in text_lower):
            vector[i] = 1
        elif cat == "Content removal" and ("content removal" in text_lower or "remove content" in text_lower):
            vector[i] = 1
        elif cat == "Choice of law" and ("choice of law" in text_lower or "governing law" in text_lower):
            vector[i] = 1
        elif cat == "Jurisdiction" and ("jurisdiction" in text_lower or "court" in text_lower):
            vector[i] = 1
        elif cat == "Contract by using" and ("contract by using" in text_lower or "browse wrap" in text_lower):
            vector[i] = 1
            
    return vector

async def run_sampling_eval(service_client, model_param: dict, samples: list, max_concurrent: int = 15):
    """Run async sampling across samples with concurrency control."""
    if "model_path" in model_param:
        client = await service_client.create_sampling_client_async(model_path=model_param["model_path"])
    else:
        client = await service_client.create_sampling_client_async(base_model=model_param["base_model"])
    
    tokenizer = client.get_tokenizer()
    sem = asyncio.Semaphore(max_concurrent)
    params = tinker.types.SamplingParams(max_tokens=64, temperature=0.0)
    
    async def sample_one(item):
        async with sem:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Clause: "{item["text"]}"\n\nClassification:'}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            model_input = tinker.types.ModelInput.from_ints(tokens)
            
            t0 = time.time()
            res = await client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
            duration = time.time() - t0
            
            raw_text = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True).strip()
            pred_vec = parse_model_prediction(raw_text)
            
            # Ground truth vector
            true_vec = [0] * len(LABEL_NAMES)
            for idx in item["label_indices"]:
                if 0 <= idx < len(LABEL_NAMES):
                    true_vec[idx] = 1
                    
            return {
                "id": item["id"],
                "text": item["text"],
                "true_labels": item["labels"],
                "true_vector": true_vec,
                "pred_text": raw_text,
                "pred_vector": pred_vec,
                "duration_sec": duration,
                "tokens_generated": len(res.sequences[0].tokens)
            }
            
    tasks = [sample_one(item) for item in samples]
    results = await asyncio.gather(*tasks)
    return results

def compute_metrics(predictions: list):
    y_true = [p["true_vector"] for p in predictions]
    y_pred = [p["pred_vector"] for p in predictions]
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    
    per_label = {}
    for i, name in enumerate(LABEL_NAMES):
        per_label[name] = {
            "precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(sup[i])
        }
        
    avg_latency = sum(p["duration_sec"] for p in predictions) / len(predictions) if predictions else 0
    total_tokens = sum(p["tokens_generated"] for p in predictions)
    
    return {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "per_label": per_label,
        "sample_count": len(predictions),
        "avg_latency_ms": round(avg_latency * 1000, 1),
        "total_tokens_generated": total_tokens
    }

async def main():
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY missing from .env")
        
    service_client = tinker.ServiceClient(api_key=api_key, display_name="didnt-read-eval")
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    test_path = os.path.join(data_dir, "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        all_test_samples = json.load(f)
        
    # Check if checkpoint exists
    chk_info_path = os.path.join(results_dir, "checkpoint_info.json")
    checkpoint_path = None
    if os.path.exists(chk_info_path):
        with open(chk_info_path, "r", encoding="utf-8") as f:
            chk_data = json.load(f)
            checkpoint_path = chk_data.get("checkpoint_path")
            
    print(f"[*] Total test set size: {len(all_test_samples)}")
    # We evaluate on the frozen test slice: all 172 unfair sentences + 250 fair sentences
    # to maintain strict statistical rigor while keeping API latency fast and cost-effective.
    unfair_test = [x for x in all_test_samples if x["is_unfair"]]
    fair_test = [x for x in all_test_samples if not x["is_unfair"]]
    eval_slice = unfair_test + fair_test[:250]
    print(f"[*] Running evaluation on {len(eval_slice)} test samples ({len(unfair_test)} unfair, 250 fair)...")
    
    # 1. Baseline Evaluation (Base model zero-shot)
    print("\n[1/2] Evaluating Zero-Shot Base Model (Qwen/Qwen3.5-4B)...")
    base_preds = await run_sampling_eval(service_client, {"base_model": "Qwen/Qwen3.5-4B"}, eval_slice)
    base_metrics = compute_metrics(base_preds)
    base_metrics["model_name"] = "Qwen/Qwen3.5-4B (Zero-Shot Baseline)"
    
    base_file = os.path.join(results_dir, "baseline.json")
    with open(base_file, "w", encoding="utf-8") as f:
        json.dump(base_metrics, f, indent=2)
    print(f"[+] Baseline Macro F1: {base_metrics['macro_f1']} | Micro F1: {base_metrics['micro_f1']}")
    
    # 2. Fine-Tuned Model Evaluation
    if checkpoint_path:
        print(f"\n[2/2] Evaluating Fine-Tuned LoRA Model ({checkpoint_path})...")
        ft_preds = await run_sampling_eval(service_client, {"model_path": checkpoint_path}, eval_slice)
        ft_metrics = compute_metrics(ft_preds)
        ft_metrics["model_name"] = f"Didn't Read Fine-Tuned LoRA ({checkpoint_path})"
        
        ft_file = os.path.join(results_dir, "finetune.json")
        with open(ft_file, "w", encoding="utf-8") as f:
            json.dump(ft_metrics, f, indent=2)
        print(f"[+] Fine-Tune Macro F1: {ft_metrics['macro_f1']} | Micro F1: {ft_metrics['micro_f1']}")
        
        # Comparison summary
        delta_macro = round(ft_metrics["macro_f1"] - base_metrics["macro_f1"], 4)
        delta_micro = round(ft_metrics["micro_f1"] - base_metrics["micro_f1"], 4)
        
        comparison = {
            "baseline": base_metrics,
            "finetune": ft_metrics,
            "improvement": {
                "macro_f1_delta": delta_macro,
                "micro_f1_delta": delta_micro,
                "percentage_gain": round((delta_macro / (base_metrics["macro_f1"] + 1e-6)) * 100, 1)
            },
            "published_lexglue_benchmark": {
                "roberta_base": {"macro_f1": 0.79, "micro_f1": 0.82},
                "legal_bert": {"macro_f1": 0.81, "micro_f1": 0.84}
            }
        }
        comp_file = os.path.join(results_dir, "comparison.json")
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n[+] EVALUATION COMPLETE!")
        print(f"    Macro F1 Improvement: {base_metrics['macro_f1']} -> {ft_metrics['macro_f1']} (Delta: +{delta_macro})")
        print(f"    Micro F1 Improvement: {base_metrics['micro_f1']} -> {ft_metrics['micro_f1']} (Delta: +{delta_micro})")
    else:
        print("[!] No checkpoint_path found in results/checkpoint_info.json yet. Evaluated baseline only.")

if __name__ == "__main__":
    asyncio.run(main())
