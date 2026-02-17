# Fine-tune Llama 3.2 1B with QLoRA on Finance Q&A

## Context
Fine-tune `meta-llama/Llama-3.2-1B` using HuggingFace SFTTrainer with QLoRA (4-bit) on the `gbharti/finance-alpaca` dataset (~68k examples). The goal is a finance-specialized model that can be used locally via Ollama.

## Prerequisites
- Accept the Llama 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-1B
- Create a HuggingFace token and add `HF_TOKEN=hf_...` to `.env`

## Files to Create/Modify

| File | Action |
|------|--------|
| `sft_finetune.py` | **Create** - main training script |
| `requirements.txt` | **Update** - add new dependencies |
| `.gitignore` | **Update** - add model output dirs |

## Step 1: Update `requirements.txt`

Add: `transformers>=4.45.0`, `datasets`, `peft>=0.13.0`, `bitsandbytes>=0.43.0`, `trl>=0.12.0`, `accelerate`, `huggingface_hub`, `python-dotenv`

Then run `pip install -r requirements.txt`

## Step 2: Update `.gitignore`

Add entries for `results/`, `finetuned-llama-finance/`, `merged-llama-finance/`, `*.gguf`

## Step 3: Create `sft_finetune.py`

The script has these sections:

### 3a. Auth & Imports
- Load `HF_TOKEN` from `.env`, call `huggingface_hub.login()`

### 3b. Load Model in 4-bit
- `BitsAndBytesConfig`: `load_in_4bit=True`, `nf4` quant type, `bfloat16` compute dtype, double quant enabled
- `AutoModelForCausalLM.from_pretrained()` with `device_map="auto"`, `attn_implementation="sdpa"`

### 3c. Tokenizer
- `AutoTokenizer.from_pretrained()`, set `pad_token = eos_token`, `padding_side = "right"`

### 3d. Dataset Loading & Formatting
- Load `gbharti/finance-alpaca` (split="train")
- Format each row into Llama 3.2 chat template using `tokenizer.apply_chat_template()`:
  - System: "You are a knowledgeable financial assistant..."
  - User: instruction + optional input context
  - Assistant: output
- Train/test split: 95/5

### 3e. LoRA Config
- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Call `prepare_model_for_kbit_training(model)` before applying LoRA

### 3f. Training Arguments
- `num_train_epochs=1`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` (effective batch=16)
- `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`
- `optim="paged_adamw_8bit"`, `bf16=True`, `gradient_checkpointing=True`
- `max_grad_norm=0.3`, `save_total_limit=2`, `output_dir="./results"`

### 3g. SFTTrainer & Train
- `SFTTrainer` with `max_seq_length=512`, `dataset_text_field="text"`, `packing=False`
- Call `trainer.train()`

### 3h. Save Adapter
- `trainer.save_model("./finetuned-llama-finance")`

### 3i. Merge & Test Inference
- Reload base model in bf16, load adapter with `PeftModel`, call `merge_and_unload()`
- Save merged model to `./merged-llama-finance`
- Run a test prompt: "What is dollar cost averaging?"

### 3j. (Optional) Ollama Integration
- Convert merged model to GGUF using llama.cpp `convert_hf_to_gguf.py` with `q4_k_m`
- Create Ollama `Modelfile` and run `ollama create llama-finance`
- Then usable via litellm as `model="ollama/llama-finance"`

## Verification
1. Run `pip install -r requirements.txt` - all packages install without errors
2. Run `python sft_finetune.py` - training starts, loss decreases over steps
3. After training: adapter saved to `./finetuned-llama-finance/`
4. Test inference produces coherent finance answers
5. (Optional) `ollama run llama-finance` works for local chat

## Notes
- **VRAM**: ~4-5GB during training (fits 6GB GPU)
- **Training time**: ~2.5-3.5 hours for 1 epoch on laptop GPU
- **If OOM**: reduce `max_seq_length` to 256 or `batch_size` to 1
- **bitsandbytes on Windows**: v0.43+ has native Windows wheels; if issues, try WSL2
