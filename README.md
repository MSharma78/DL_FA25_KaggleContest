# LLaMA-3 LoRA for Math Solution Verification

This repo contains our end-to-end pipeline for the *NYU DL* Kaggle competition.
We fine-tune **meta-llama/Meta-Llama-3-8B** in 4-bit with **LoRA**, validate using a **forced-choice decoder** (compare the model’s likelihood of the tokens “True” vs “False”), export the best adapter, and produce `submission_final.csv`.

**Best public LB so far:** **0.82518**
**Hardware used:** NVIDIA A100 (bf16)

---

## 📦 Best-adapter folder (download)

GitHub can’t host large binaries easily, so the full **`best-adapter/`** directory (including the large `.safetensors` file) is stored on Google Drive:

**Download folder:**
[https://drive.google.com/drive/folders/1Em9LdMd2pCsrDPk-IGtT2zrTblnjRcp9?usp=sharing](https://drive.google.com/drive/folders/1Em9LdMd2pCsrDPk-IGtT2zrTblnjRcp9?usp=sharing)

After downloading, place the folder in the repo root so the structure looks like:

```text
DL_FA25_KAGGLECONTEST/
  best-adapter/
    adapter_config.json
    adapter_model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
  rpm_dl_midterm.ipynb
  submission_final.csv
  README.md
```

If you already have a `best-adapter/` directory from the repo, you can simply **overwrite it** with the downloaded one to ensure `adapter_model.safetensors` is present.

---

## Repository layout

* **best-adapter/**

  * `adapter_config.json` — PEFT/LoRA configuration
  * `adapter_model.safetensors` — **adapter weights (required; from Drive)**
  * `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json` — tokenizer metadata
* **rpm_dl_midterm.ipynb** — notebook to train, validate, export, and infer
* **submission_final.csv** — latest competition submission
* **README.md** — this file

---

## Data

* Dataset: **`ad6398/nyu-dl-teach-maths-comp`** (train + test splits on Hugging Face).
* Typical splits used here:

  * Train subset: first *N* examples after shuffling with a fixed seed.
  * Validation subset: next *M* examples (held out).
  * Test: full 10,000 examples.

---

## Training pipeline (high level)

1. **Model & quantization:** load Meta-LLaMA-3-8B in 4-bit (bitsandbytes) with bf16 compute; Flash-Attention 2 if available.
2. **Prompting:** supervised fine-tuning with an instruction template that shows the Question, the proposed Solution, and the target Output as either `True` or `False`.
3. **LoRA:** apply parameter-efficient fine-tuning (attention + MLP adapters in our best run).
4. **Sequence packing:** pack multiple samples per sequence to increase tokens/sec.
5. **Optimization:** AdamW (8-bit), cosine schedule with warmup, regular validation and checkpoint saving.
6. **Model selection:** keep `load_best_model_at_end` and select by lowest validation loss.
7. **Export:** save the best LoRA adapter to `best-adapter/` (small artifact), reusing the tokenizer metadata.
8. **Validation decoding:** evaluate with forced-choice (True vs False) and **tune a scalar decision bias** on validation to maximize accuracy.
9. **Test inference:** run on the test split and write `submission_final.csv`.

---

## Inference strategy

* **Attach the adapter** from `best-adapter/` to the 4-bit LLaMA-3 base model.
* Use **forced-choice decoding**: compute the model’s likelihood for the literal strings **“True”** and **“False”** given the prompt context; pick the higher one.
* **Bias tuning:** sweep a small scalar offset on the decision margin (difference in log-likelihoods) over a validation set and fix the best value; this typically yields **+0.3 to +1.5 pp**.

---

## Reproducing `submission_final.csv`

1. Train with regular evaluation and `load_best_model_at_end`.
2. Export the best adapter to `best-adapter/` (or download it from Drive).
3. Validate, tune the decision bias, then run on the test split.
4. Save a two-column CSV: `ID` (0..9999), `is_correct` (True/False).

---

## Environment (tested)

* Transformers **≥4.44, <4.47**
* TRL **≥0.9.0, <0.11**
* PEFT **≥0.11, <0.13**
* Accelerate **≥0.33, <0.36**
* Datasets **≥2.19**
* BitsAndBytes **≥0.44**
* (Optional) Flash-Attention **≥2.5.6** for large speedups on A100

**Common fixes**

* TRL/Trainer signature errors around `tokenizer` / `max_seq_length`: remove those constructor args or upgrade TRL into the range above.
* `eval_strategy=steps` but no eval dataset: either pass an eval split or set evaluation to “no”.
* `AcceleratorState ... distributed_type`: restart the runtime/kernel before creating a new trainer to clear state.

---

## License

* Base model: **Meta-LLaMA-3-8B** (Meta license applies).
* This repository and adapter are for educational competition use.
