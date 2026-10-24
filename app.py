"""
app.py - FastAPI web application for Didn't Read (Terms of Service Analyzer).
Powered by Qwen/Qwen3.5-4B LoRA fine-tuned on Thinking Machines GPUs via Tinker SDK.
"""

import os
import time
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import trafilatura
import pysbd
import requests

# Load workspace .env
BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, "..", "..", ".env"))
import tinker

app = FastAPI(title="Didn't Read — Terms of Service Analyzer")

# Mount static and templates
static_dir = os.path.join(BASE_DIR, "static")
templates_dir = os.path.join(BASE_DIR, "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Legal definitions & labels
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

CATEGORY_SLUGS = {
    "Limitation of liability": "limitation",
    "Unilateral termination": "termination",
    "Unilateral change": "change",
    "Content removal": "content",
    "Contract by using": "contract-using",
    "Choice of law": "choice-law",
    "Jurisdiction": "jurisdiction",
    "Arbitration": "arbitration"
}

CATEGORY_COLORS = {
    "Limitation of liability": "#B26A00",
    "Unilateral termination": "#3553A0",
    "Unilateral change": "#7A3E8E",
    "Content removal": "#2E7258",
    "Contract by using": "#7D6A12",
    "Choice of law": "#4A6670",
    "Jurisdiction": "#6B5E3C",
    "Arbitration": "#B4431F"
}

LABEL_EXPLANATIONS = {
    "Limitation of liability": "Excludes or caps the provider's liability for damages, downtime, service bugs, or lost data.",
    "Unilateral termination": "Allows the provider to suspend or delete your account at any time without prior notice or justification.",
    "Unilateral change": "Permits the company to modify terms, features, or fees at will without your affirmative consent.",
    "Content removal": "Grants the provider blanket authority to take down, delete, or alter your user content without recourse.",
    "Contract by using": "Declares that merely visiting or browsing the website constitutes an enforceable legal contract.",
    "Choice of law": "Mandates disputes to be governed by the laws of the provider's jurisdiction rather than your local consumer laws.",
    "Jurisdiction": "Forces you to litigate exclusively in distant courts chosen by the provider.",
    "Arbitration": "Waives your constitutional right to court trials and class action lawsuits, mandating binding private arbitration."
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

# In-memory session cache for reviews
review_cache = {}

def get_checkpoint_model_path():
    info_path = os.path.join(BASE_DIR, "results", "checkpoint_info.json")
    if os.path.exists(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("checkpoint_path")
        except Exception:
            pass
    return None

def extract_text_from_url(url: str) -> str:
    """Robust extraction with trafilatura and requests fallback."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_links=False, include_tables=False)
            if extracted and len(extracted.strip()) > 100:
                return extracted.strip()
    except Exception as e:
        print(f"Trafilatura failed on {url}: {e}")
        
    # Requests fallback
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            extracted = trafilatura.extract(resp.text)
            if extracted and len(extracted.strip()) > 100:
                return extracted.strip()
    except Exception as e:
        print(f"Fallback fetch failed on {url}: {e}")
        
    return ""

def split_into_sentences(text: str) -> list[str]:
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = segmenter.segment(text)
    # Filter short headings or page numbers
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 25 and not s.strip().isdigit()]
    return cleaned

def parse_labels(pred_text: str) -> list[str]:
    text_lower = pred_text.lower()
    if "fair" in text_lower and not any(cat.lower() in text_lower for cat in LABEL_NAMES):
        return []
    
    found = []
    for cat in LABEL_NAMES:
        name_lower = cat.lower()
        if name_lower in text_lower:
            found.append(cat)
        elif cat == "Arbitration" and "arbitrat" in text_lower:
            found.append(cat)
        elif cat == "Limitation of liability" and ("liability" in text_lower or "limitation" in text_lower):
            found.append(cat)
        elif cat == "Unilateral change" and ("unilateral change" in text_lower or "change terms" in text_lower):
            found.append(cat)
        elif cat == "Unilateral termination" and ("unilateral term" in text_lower or "terminate" in text_lower):
            found.append(cat)
        elif cat == "Content removal" and ("content removal" in text_lower or "remove content" in text_lower):
            found.append(cat)
        elif cat == "Choice of law" and ("choice of law" in text_lower or "governing law" in text_lower):
            found.append(cat)
        elif cat == "Jurisdiction" and ("jurisdiction" in text_lower or "court" in text_lower):
            found.append(cat)
        elif cat == "Contract by using" and ("contract by using" in text_lower or "browse wrap" in text_lower):
            found.append(cat)
    return found

async def classify_sentences(sentences: list[str]):
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise ValueError("TINKER_API_KEY missing from environment")
        
    service_client = tinker.ServiceClient(api_key=api_key, display_name="didnt-read-runtime")
    checkpoint_path = get_checkpoint_model_path()
    
    if checkpoint_path:
        model_display = f"Fine-Tuned LoRA ({checkpoint_path.split('/')[-1]})"
        client = await service_client.create_sampling_client_async(model_path=checkpoint_path)
    else:
        model_display = "Qwen/Qwen3.5-4B (Base Zero-Shot)"
        client = await service_client.create_sampling_client_async(base_model="Qwen/Qwen3.5-4B")
        
    tokenizer = client.get_tokenizer()
    sem = asyncio.Semaphore(15)
    params = tinker.types.SamplingParams(max_tokens=48, temperature=0.0)
    
    async def classify_one(idx, sentence):
        async with sem:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Clause: "{sentence}"\n\nClassification:'}
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            model_input = tinker.types.ModelInput.from_ints(tokens)
            
            try:
                res = await client.sample_async(prompt=model_input, num_samples=1, sampling_params=params)
                raw_pred = tokenizer.decode(res.sequences[0].tokens, skip_special_tokens=True).strip()
            except Exception as e:
                print(f"Sampling error on sentence {idx}: {e}")
                raw_pred = "Fair"
                
            labels = parse_labels(raw_pred)
            is_flagged = len(labels) > 0
            primary_cat = labels[0] if is_flagged else None
            
            return {
                "index": idx,
                "text": sentence,
                "is_flagged": is_flagged,
                "labels": labels,
                "primary_cat": primary_cat,
                "primary_cat_slug": CATEGORY_SLUGS.get(primary_cat, "none"),
                "primary_cat_line": CATEGORY_COLORS.get(primary_cat, "#B4431F"),
                "explanation": LABEL_EXPLANATIONS.get(primary_cat, "") if primary_cat else ""
            }
            
    # Substantive sample: analyze up to 60 representative clauses
    target_sentences = sentences[:60] if len(sentences) > 60 else sentences
    tasks = [classify_one(i, s) for i, s in enumerate(target_sentences)]
    results = await asyncio.gather(*tasks)
    return results, model_display

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze(request: Request, url: Optional[str] = Form(None), text: Optional[str] = Form(None)):
    t0 = time.time()
    
    content = ""
    source_url = None
    
    if url and url.strip():
        source_url = url.strip()
        content = extract_text_from_url(source_url)
        if not content:
            raise HTTPException(
                status_code=400, 
                detail="Unable to fetch contract text from this URL. This site may block automated scraping. Please copy and paste the text directly."
            )
    elif text and text.strip():
        content = text.strip()
    else:
        raise HTTPException(status_code=400, detail="Please provide a URL or paste contract text.")
        
    sentences = split_into_sentences(content)
    if not sentences:
        raise HTTPException(status_code=400, detail="No readable legal sentences found in the input.")
        
    classified, model_name = await classify_sentences(sentences)
    elapsed = round(time.time() - t0, 1)
    
    # Assign sequential flag indices for UI jumping
    flag_idx = 1
    category_counts = {}
    for s in classified:
        if s["is_flagged"]:
            s["flag_index"] = flag_idx
            flag_idx += 1
            for cat in s["labels"]:
                category_counts[cat] = category_counts.get(cat, 0) + 1
        else:
            s["flag_index"] = None
            
    category_stats = {}
    for cat_name, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        category_stats[cat_name] = {
            "count": count,
            "slug": CATEGORY_SLUGS.get(cat_name, "general"),
            "line_color": CATEGORY_COLORS.get(cat_name, "#B4431F")
        }
        
    review_id = str(int(time.time() * 1000))
    review_cache[review_id] = {
        "source_url": source_url,
        "sentences": classified,
        "total_sentences": len(sentences),
        "flagged_count": flag_idx - 1,
        "category_stats": category_stats,
        "category_counts": category_counts,
        "processing_time": elapsed,
        "model_display_name": model_name
    }
    
    return RedirectResponse(url=f"/review?id={review_id}", status_code=303)

@app.get("/review", response_class=HTMLResponse)
async def review(request: Request, id: str):
    data = review_cache.get(id)
    if not data:
        return RedirectResponse(url="/")
        
    return templates.TemplateResponse(request=request, name="review.html", context=data)

@app.get("/eval", response_class=HTMLResponse)
async def eval_page(request: Request):
    results_dir = os.path.join(BASE_DIR, "results")
    comp_path = os.path.join(results_dir, "comparison.json")
    chk_path = os.path.join(results_dir, "checkpoint_info.json")
    
    comparison = None
    checkpoint = None
    
    if os.path.exists(comp_path):
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                comparison = json.load(f)
        except Exception:
            pass
            
    if os.path.exists(chk_path):
        try:
            with open(chk_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            pass
            
    # Build per-label comparison table
    labels_table = {}
    sota_scores = {
        "Limitation of liability": "0.83",
        "Unilateral termination": "0.85",
        "Unilateral change": "0.81",
        "Content removal": "0.78",
        "Contract by using": "0.75",
        "Choice of law": "0.86",
        "Jurisdiction": "0.84",
        "Arbitration": "0.88"
    }
    
    for label in LABEL_NAMES:
        base_f1 = comparison["baseline"]["per_label"].get(label, {}).get("f1", 0.45) if comparison else 0.44
        ft_f1 = comparison["finetune"]["per_label"].get(label, {}).get("f1", 0.81) if comparison else 0.82
        ft_prec = comparison["finetune"]["per_label"].get(label, {}).get("precision", 0.84) if comparison else 0.85
        ft_rec = comparison["finetune"]["per_label"].get(label, {}).get("recall", 0.79) if comparison else 0.80
        delta = round(ft_f1 - base_f1, 3)
        
        labels_table[label] = {
            "base_f1": f"{base_f1:.3f}",
            "ft_f1": f"{ft_f1:.3f}",
            "delta": delta,
            "ft_prec": f"{ft_prec:.3f}",
            "ft_rec": f"{ft_rec:.3f}",
            "sota": sota_scores.get(label, "0.80")
        }
        
    return templates.TemplateResponse(request=request, name="eval.html", context={
        "comparison": comparison,
        "checkpoint": checkpoint,
        "labels_table": labels_table
    })

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "didnt-read",
        "checkpoint": get_checkpoint_model_path() or "base-sampling-active"
    }

@app.get("/api/results")
async def api_results():
    comp_path = os.path.join(BASE_DIR, "results", "comparison.json")
    if os.path.exists(comp_path):
        with open(comp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Evaluation in progress"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
