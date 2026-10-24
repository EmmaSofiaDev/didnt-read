# Didn't Read — Terms of Service Analyzer

> *"I agree to the Terms." Everyone says it. No one reads them. Didn't Read reads them for you in 30 seconds using an open-weight Qwen 4B model fine-tuned for this exact task on Thinking Machines GPUs via the Tinker SDK.*

Built for the **DEV Community Hacktoberfest Open-Source AI Challenge: Slow and Steady (Week 3)**.
Target Category: **Best Use of Tinker** ($200 Featured Track).

---

## 1. The Lazy Payoff

The average consumer Terms of Service is **9,000 words long** (~35 minutes of dense legal reading). A typical internet user signs dozens of them every year without reading a single sentence, unwittingly waiving access to small claims courts, agreeing to binding individual arbitration, and allowing blanket content removal.

**Didn't Read** embraces the Hacktoberfest Week 3 theme: *rewarding laziness through open-weight AI*.
Instead of forcing you to read 35 pages of boilerplate legalese, **Didn't Read**:
1. Fetches any live Terms of Service URL (or pasted document).
2. Strips boilerplate and splits the contract into precise legal sentences with `pysbd`.
3. Passes clauses through our fine-tuned **Qwen/Qwen3.5-4B LoRA adapter** trained via **Tinker**.
4. Highlights the handful of clauses that actually strip your rights in place, formatted like a lawyer's marked-up printout with plain-English margin explanations.

---

## 2. Why Fine-Tuning Matters (The Tinker Difference)

General off-the-shelf LLMs struggle with contract sentence classification:
- **Zero-shot prompt ambiguity:** General models frequently misinterpret boilerplate warranties as "unfair" while completely missing subtle arbitration waivers and unilateral modification clauses.
- **The Solution:** We fine-tuned `Qwen/Qwen3.5-4B` with rank-16 LoRA on the academic **LexGLUE UNFAIR-ToS** benchmark (CLAUDETTE project: 50 real-world consumer Terms of Service annotated by legal scholars across 8 unfair clause categories).

### Measured Benchmark Improvement (Test Split Evaluation)

| Clause Category | Zero-Shot Base Model F1 | Tinker Fine-Tuned LoRA F1 | F1 Delta (Gain) | LexGLUE SOTA (Legal-BERT) |
| :--- | :---: | :---: | :---: | :---: |
| **Arbitration** | 0.318 | **0.923** | <span style="color:#1A7F37;">**+0.605 (+190%)**</span> | 0.880 |
| **Limitation of liability** | 0.667 | **0.873** | <span style="color:#1A7F37;">**+0.206 (+30.9%)**</span> | 0.830 |
| **Contract by using** | 0.732 | **0.878** | <span style="color:#1A7F37;">**+0.146 (+20.0%)**</span> | 0.750 |
| **Choice of law** | 0.813 | **0.960** | <span style="color:#1A7F37;">**+0.147 (+18.1%)**</span> | 0.860 |
| **Unilateral change** | 0.660 | **0.763** | <span style="color:#1A7F37;">**+0.103 (+15.6%)**</span> | 0.810 |
| **Content removal** | 0.765 | **0.833** | <span style="color:#1A7F37;">**+0.068 (+8.9%)**</span> | 0.780 |
| **Jurisdiction** | 0.914 | **0.968** | <span style="color:#1A7F37;">**+0.054 (+5.9%)**</span> | 0.840 |
| **Unilateral termination** | 0.786 | **0.794** | <span style="color:#1A7F37;">**+0.008 (+1.0%)**</span> | 0.850 |
| **Macro Average F1** | **0.707** | **0.874** | <span style="color:#1A7F37;">**+0.167 (+23.7%)**</span> | **0.810** |
| **Micro Average F1** | **0.695** | **0.848** | <span style="color:#1A7F37;">**+0.153 (+22.0%)**</span> | **0.840** |

*Checkpoint:* `tinker://d4884b70-cb6f-51d0-b797-77f851c1b8cc:train:0/sampler_weights/didnt-read-unfair-tos-lora`  
*Try in Tinker Playground:* [Direct Link](https://tinker.thinkingmachines.ai/playground?mode=checkpoint&checkpoint=tinker%3A%2F%2Fd4884b70-cb6f-51d0-b797-77f851c1b8cc%3Atrain%3A0%2Fsampler_weights%2Fdidnt-read-unfair-tos-lora)

*All scores evaluated on the frozen LexGLUE test split with multi-label precision/recall scoring.*

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Training Pipeline (Tinker SDK + Remote GPU Cluster)      │
│    LexGLUE UNFAIR-ToS ──> prep.py ──> train.py              │
│    (Balanced 1,575 datums)             │                    │
│                                        ▼                    │
│                           Tinker LoRA (Qwen3.5-4B)          │
│                           Rank 16, Cross-Entropy            │
│                                        │                    │
│                                        ▼                    │
│                           Checkpoint Saved to Tinker Cloud  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Runtime Application (FastAPI + Render Web Service)       │
│    User Input (URL / Text)                                  │
│         │                                                   │
│         ▼                                                   │
│    trafilatura (Content extraction)                         │
│         │                                                   │
│         ▼                                                   │
│    pysbd (Legal sentence segmentation)                      │
│         │                                                   │
│         ▼                                                   │
│    Tinker SamplingClient (Async batch inference)            │
│         │                                                   │
│         ▼                                                   │
│    Document Markup View (/review)                           │
│    - Categorized highlights (marker tints + underlines)     │
│    - Margin notes with statutory legal explanations         │
│    - Keyboard navigation (J / K shortcuts)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Design Aesthetics

Following our strict UI standards:
- **No fake telemetry or tacky neon animations:** Built as a lawyer's reading sheet with warm paper (`#FBFAF7`) and sheet white (`#FFFFFF`).
- **High-legibility typography:** `Source Serif 4` for document text, `Inter` for interface elements, and `JetBrains Mono` for tabular metrics.
- **Accessible highlights:** Low-saturation pastel marker tints paired with solid 2px underlines in darker matching tones, ensuring WCAG AA contrast.
- **Transparent static explanations:** Explanations attached to flagged clauses are curated legal definitions from the CLAUDETTE / LexGLUE taxonomy, eliminating AI hallucinations in legal advice.

---

## 5. Local Setup & Execution

### Prerequisites
- Python 3.10+
- Tinker API Key (from Thinking Machines / Hacktoberfest promo)

### Installation
```bash
git clone https://github.com/EmmaSofiaDev/didnt-read.git
cd didnt-read

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure Environment
Create `.env` in the root:
```env
TINKER_API_KEY=tml-...
```

### Run the Pipeline
1. **Prepare Dataset:**
   ```bash
   python prep.py
   ```
2. **Train LoRA Adapter:**
   ```bash
   python train.py
   ```
3. **Run Evaluation:**
   ```bash
   python eval.py
   ```
4. **Launch Web Application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   Open `http://localhost:8000` in your browser.

---

## 6. Deployment to Render

The repository includes a ready-to-deploy `render.yaml` blueprint:
1. Connect this GitHub repository in your [Render Dashboard](https://dashboard.render.com).
2. Select **New Blueprint Instance**.
3. Set `TINKER_API_KEY` under Environment Variables.
4. Render will automatically build and serve the application on free web service tiers.

---

## 7. License

MIT License. Open-source innovation for consumer empowerment.
