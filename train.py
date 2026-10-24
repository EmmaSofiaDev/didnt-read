"""
train.py - Fine-tune Qwen/Qwen3.5-4B LoRA using Tinker for UNFAIR-ToS classification.
Trains on Thinking Machines' remote GPU infrastructure via Tinker SDK.
"""

import os
import json
import asyncio
import random
import time
from dotenv import load_dotenv

# Load env from Dev2 root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
import tinker

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

def make_datum(tokenizer, text: str, target: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Clause: "{text}"\n\nClassification:'},
        {"role": "assistant", "content": target}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_tokens = tokenizer.encode(
        full_text[len(prompt_text):], add_special_tokens=False
    )
    tokens = prompt_tokens + completion_tokens
    # Train only on completion tokens (the classification answer)
    weights = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)
    
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={"target_tokens": tokens[1:], "weights": weights[1:]}
    )

async def main():
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY is missing from .env")
    
    print("[*] Initializing Tinker ServiceClient...")
    service_client = tinker.ServiceClient(
        api_key=api_key,
        display_name="didnt-read-lora-training"
    )
    
    base_model = "Qwen/Qwen3.5-4B"
    lora_rank = 16
    print(f"[*] Creating LoRA Training Client: base_model={base_model}, rank={lora_rank}...")
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model, rank=lora_rank
    )
    tokenizer = training_client.get_tokenizer()
    print("[+] Training client connected and tokenizer loaded.")
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_path = os.path.join(data_dir, "train_balanced.json")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}. Run prep.py first!")
    
    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = json.load(f)
    
    print(f"[*] Loaded {len(train_samples)} training samples.")
    
    # Prepare Datums
    print("[*] Tokenizing training samples into Tinker Datums...")
    datums = []
    for item in train_samples:
        d = make_datum(tokenizer, item["text"], item["target"])
        datums.append(d)
    print(f"[+] Prepared {len(datums)} Datums.")
    
    # Hyperparameters
    batch_size = 16
    epochs = 2
    steps_per_epoch = len(datums) // batch_size
    total_steps = steps_per_epoch * epochs
    learning_rate = 2e-4
    
    print(f"[*] Starting Training: {total_steps} steps (batch_size={batch_size}, lr={learning_rate})...")
    
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    training_history = []
    start_time = time.time()
    
    step = 0
    for epoch in range(epochs):
        random.seed(42 + epoch)
        indices = list(range(len(datums)))
        random.shuffle(indices)
        
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            step += 1
            batch_indices = indices[i : i + batch_size]
            batch_data = [datums[idx] for idx in batch_indices]
            
            # Forward + Backward and Optim Step
            fwdbwd_future = await training_client.forward_backward_async(batch_data, "cross_entropy")
            optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=learning_rate))
            
            fwdbwd_result = await fwdbwd_future.result_async()
            await optim_future.result_async()
            
            # Compute loss
            logprobs = [
                lp
                for output in fwdbwd_result.loss_fn_outputs
                for lp in output["logprobs"].tolist()
            ]
            weights = [
                w for d in batch_data for w in d.loss_fn_inputs["weights"].tolist()
            ]
            total_weight = sum(weights)
            loss = -sum(lp * w for lp, w in zip(logprobs, weights)) / total_weight if total_weight > 0 else 0.0
            
            record = {
                "step": step,
                "epoch": epoch + 1,
                "loss": round(loss, 4),
                "timestamp": round(time.time() - start_time, 2)
            }
            training_history.append(record)
            
            if step % 5 == 0 or step == total_steps or step == 1:
                print(f"[Step {step:3d}/{total_steps}] Epoch {epoch+1} | Loss: {loss:.4f} | Elapsed: {record['timestamp']}s")
    
    total_duration = time.time() - start_time
    print(f"[+] Training completed in {total_duration:.1f}s!")
    
    # Save checkpoint for sampler
    checkpoint_name = "didnt-read-unfair-tos-lora"
    print(f"[*] Saving weights for sampler as '{checkpoint_name}'...")
    save_future = await training_client.save_weights_for_sampler_async(name=checkpoint_name)
    checkpoint = await save_future.result_async()
    
    playground_url = checkpoint.get_playground_url() if hasattr(checkpoint, "get_playground_url") else ""
    checkpoint_path = getattr(checkpoint, "path", str(checkpoint))
    print(f"[+] Checkpoint saved successfully!")
    print(f"    Path: {checkpoint_path}")
    if playground_url:
        print(f"    Playground: {playground_url}")
    
    checkpoint_info = {
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": checkpoint_path,
        "base_model": base_model,
        "rank": lora_rank,
        "total_steps": total_steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": training_history[-1]["loss"] if training_history else None,
        "training_time_seconds": round(total_duration, 2),
        "history": training_history
    }
    
    info_path = os.path.join(results_dir, "checkpoint_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_info, f, indent=2)
    print(f"[+] Saved checkpoint metadata to {info_path}")
    
    # Save loss curve
    loss_path = os.path.join(results_dir, "training_loss.json")
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    print(f"[+] Saved training history to {loss_path}")

if __name__ == "__main__":
    asyncio.run(main())
