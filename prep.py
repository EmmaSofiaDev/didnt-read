"""
prep.py - Download and prepare the LexGLUE UNFAIR-ToS dataset.
Formats data for Tinker LoRA training, zero-shot baseline, and test evaluation.
"""

import os
import json
import random
from datasets import load_dataset

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

LABEL_EXPLANATIONS = {
    "Limitation of liability": "Excludes or caps the provider's liability for damages, system downtime, or financial loss.",
    "Unilateral termination": "Allows the provider to suspend or delete your account at any time without prior warning or justification.",
    "Unilateral change": "Permits the provider to alter terms, features, or fees at will without your affirmative consent.",
    "Content removal": "Grants the provider blanket discretion to delete, hide, or alter your user content without recourse.",
    "Contract by using": "Declares that simply viewing or navigating the website binds you to an enforceable legal agreement.",
    "Choice of law": "Requires legal disputes to be governed by the laws of the provider's jurisdiction rather than yours.",
    "Jurisdiction": "Forces you to submit to distant or specific court venues, barring local consumer courts.",
    "Arbitration": "Waives your constitutional right to a court trial and class action, forcing private individual arbitration."
}

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

def format_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Clause: "{text.strip()}"\n\nClassification:'}
    ]

def prepare_dataset():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    print("[*] Loading coastalcph/lex_glue ('unfair_tos')...")
    ds = load_dataset("coastalcph/lex_glue", "unfair_tos")
    
    splits_data = {}
    
    for split_name in ["train", "validation", "test"]:
        split = ds[split_name]
        items = []
        for idx, row in enumerate(split):
            text = row["text"].strip()
            label_indices = row["labels"]
            names = [LABEL_NAMES[i] for i in label_indices if 0 <= i < len(LABEL_NAMES)]
            target = ", ".join(names) if names else "Fair"
            
            items.append({
                "id": idx,
                "text": text,
                "label_indices": label_indices,
                "labels": names,
                "is_unfair": len(names) > 0,
                "target": target
            })
        
        splits_data[split_name] = items
        out_path = os.path.join(data_dir, f"{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        
        unfair_count = sum(1 for x in items if x["is_unfair"])
        fair_count = len(items) - unfair_count
        print(f"[{split_name.upper()}] Total: {len(items)} | Unfair: {unfair_count} | Fair: {fair_count} -> Saved to {out_path}")

    # Create balanced training set for LoRA fine-tuning
    # UNFAIR-ToS is heavily imbalanced (~90% fair). To ensure the model learns unfair signatures,
    # we take all unfair train samples and balance with an equal number of fair samples.
    train_items = splits_data["train"]
    unfair_train = [x for x in train_items if x["is_unfair"]]
    fair_train = [x for x in train_items if not x["is_unfair"]]
    
    random.seed(42)
    # 1:1.5 ratio of unfair to fair
    fair_sampled = random.sample(fair_train, min(len(fair_train), int(len(unfair_train) * 1.5)))
    balanced_train = unfair_train + fair_sampled
    random.shuffle(balanced_train)
    
    balanced_path = os.path.join(data_dir, "train_balanced.json")
    with open(balanced_path, "w", encoding="utf-8") as f:
        json.dump(balanced_train, f, indent=2, ensure_ascii=False)
    
    print(f"[BALANCED TRAIN] Created {len(balanced_train)} samples ({len(unfair_train)} unfair, {len(fair_sampled)} fair) -> {balanced_path}")
    print("[+] Dataset preparation script ready!")

if __name__ == "__main__":
    prepare_dataset()
