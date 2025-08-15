# Research on Chinese Medical Question Answering Model Based on Model Fine-tuning & RAG

> Fine-tune a compact instruction model (Qwen2.5-0.5B-Instruct) on Chinese MedQA, evaluate it three ways (direct, HTTP API, API + RAG), and expose a Gradio UI reachable from your local machine via SSH tunneling.

---

## 1) Overview

This repository provides an end-to-end workflow to:
- Convert raw MedQA (Simplified Chinese Subset) to **SFT** format.
- Fine-tune **Qwen2.5-0.5B-Instruct** with **LLaMA-Factory** using both **LoRA** and **full-parameter** training.
- Evaluate the models' performance via direct call or API Call(implement the trained/non-trained model as HTTP API Service).
- Add **RAG** retrieval to improve answers and re-evaluate.
- Run an interactive **Gradio** demo on AutoDL and access it locally via **SSH Tunneling Tool**.
- (Optional) Connect the model to **Dify** *(YouTube Link: https://youtu.be/qT1t545kN6I)*.

---

## 2) Repository Layout (example)

```
root/
├── autodl-tmp/
│   └── LLaMA-Factory/
│       ├──2025_05_22_med_data_zh_paragraph/
│       ├── data/
│       │   ├── process_data.py
│       │   ├── train.jsonl
│       │   ├── test.jsonl
│       │   └── dataset_info.json
│       ├── examples/
│       │   ├── train_lora/llama3_lora_sft.yaml
│       │   ├── train_full/llama3_full_sft.yaml
│       │   ├── merge_lora/
│       │   └── inference/
│       ├── evaluation/
│       │   ├── Direct Evaluation/
│       │   ├── API Evaluation/
│       │   └── API with Rag Evaluation/
│       └── Gradio_demo_new.py
└── Qwen2.5-0.5B-Instruct/
```

**Textbooks Download**: The textbooks in 'LLaMA-Factory/2025_05_22_med_data_zh_paragraph' are not complete,should be downloaded by https://github.com/jind11/MedQA?tab=readme-ov-file google drive (data_clean/data_clean/zh_paragrah)

---

## 3) Prerequisites

- **GPU**: NVIDIA with sufficient VRAM (≥4 GB recommended for full-parameter training; LoRA is lighter).
- **OS / CUDA / Python**: AutoDL (or similar) with Python 3.10+ and CUDA-compatible PyTorch.
- **Git LFS** and (optionally) **Hugging Face CLI** for model downloads.

---

## 4) External Downloads (must be prepared)

1. **Qwen2.5-0.5B-Instruct** — download from **Hugging Face** and place at:
   ```text
   ./Qwen2.5-0.5B-Instruct
   ```

2. **LLaMA-Factory** — clone and install from:
   https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git autodl-tmp/LLaMA-Factory
   cd autodl-tmp/LLaMA-Factory
   pip install -e ".[torch,metrics,quality]"
   ```

> Most commands below are executed **inside** `autodl-tmp/LLaMA-Factory` unless specified.

---

## 5) Data Preparation (Convert to SFT)

1. Put the **raw MedQA (Simplified Chinese Subset)** data under `LLaMA-Factory/data/`.
2. Run the conversion script to produce SFT format:
   ```bash
   cd autodl-tmp/LLaMA-Factory/data
   python process_data.py
   ```

This produces `train.jsonl`  in the expected SFT structure.

---

## 6) Configure Training

Edit the following YAMLs to match your environment and paths:

- **LoRA**: `examples/train_lora/llama3_lora_sft.yaml`
- **Full-parameter**: `examples/train_full/llama3_full_sft.yaml`
Please refer to the .yaml files for parameter changes/edited.
---

## 7) Training & Export

Under `autodl-tmp/LLaMA-Factory` root:

### A) LoRA Fine-tuning & Export
```bash
# Train LoRA
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# Merge / export LoRA weights
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### B) Full-parameter Fine-tuning
```bash
llamafactory-cli train examples/train_full/llama3_full_sft.yaml
# (Saved checkpoints are in the configured output directory.)
```

---

## 8) Evaluation (Three Approaches)

Your evaluation code lives under `evaluation/`:

### 8.1 Direct Evaluation (no server)
- Folder: `evaluation/Direct Evaluation`
- Example: `evaluate_base_model.py` (set `model_path` to the base or fine-tuned path in the script)
```bash
cd evaluation/Direct\ Evaluation
python evaluate_base_model.py
```

### 8.2 HTTP API Evaluation (vLLM backend)
1) Serve the model as an HTTP API **from LLaMA-Factory root**:
```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml   infer_backend=vllm vllm_enforce_eager=true
# For fine-tuned models, switch the YAML:
#   examples/inference/llama3_lora_sft.yaml
#   examples/inference/llama3_full_sft.yaml
```

2) Run the evaluating scripts:
- Folder: `evaluation/API Evaluation`
- Example: `base_API_evaluation.py` (make sure the served YAML matches the intended model)
```bash
cd evaluation/"API Evaluation"
python base_API_evaluation.py
```

### 8.3 HTTP API + RAG Evaluation
- Folder: `evaluation/API with Rag Evaluation`
- Example: `Rag_model_evaluation.py` (same model serving as 8.2, plus retrieval configs)
```bash
cd evaluation/"API with Rag Evaluation"
python Rag_model_evaluation.py
```

---

## 9) Interactive Demo (Gradio) via SSH Tunneling

1. **SSH Tunneling Tool (local machine)**  
   - Fill instance SSH command and password.  
   - Set **“Proxy to local port” = `7860`** and keep remote port (e.g., `1080`).  
2. **On the AutoDL instance** (in `LLaMA-Factory/`):
   ```bash
   python Gradio_demo_new.py
   ```
3. **Back to the tunneling tool** → Click **Start Proxy**.  
4. Open the browser at **http://127.0.0.1:7860** to access the remote Gradio UI locally.

---

## 10) Dify Integration (Optional)

This project can be connected to **Dify** by pointing Dify to the running HTTP API endpoint.  
**YouTube link:** *(https://youtu.be/qT1t545kN6I).*

---

## 11) Quickstart (Copy–Paste)

```bash
# Clone LLaMA-Factory and install
cd ~/autodl-tmp
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,quality]"

# Prepare Qwen2.5-0.5B-Instruct at: ~/autodl-tmp/Qwen2.5-0.5B-Instruct

# Convert data to SFT
cd data && python process_data.py 

# Edit YAMLs under examples/train_lora and examples/train_full

# Train & export
llamafactory-cli train  examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
llamafactory-cli train  examples/train_full/llama3_full_sft.yaml

# Serve API (choose one YAML)
API_PORT=8000 llamafactory-cli api examples/inference/llama3_full_sft.yaml infer_backend=vllm vllm_enforce_eager=true

# Evaluate
python evaluation/Direct\ Evaluation/evaluate_base_model.py
python evaluation/"API Evaluation"/base_API_evaluation.py
python evaluation/"API with Rag Evaluation"/Rag_model_evaluation.py

# Run Gradio on the server, then visit http://127.0.0.1:7860 via SSH tunneling.
```

---

## 12) Troubleshooting

- **CUDA OOM**: Reduce `micro_batch_size`, enable gradient checkpointing, or start with LoRA.  
- **API not reachable**: Ensure the `llamafactory-cli api ...` process is running on port `8000`.  
- **RAG errors**: Verify your documents directory exists and the embedder/checkpoint paths are correct.  
- **Path issues**: Double-check `model_name_or_path`, `data_path`, and `output_dir` in YAMLs.

---

## 13) License & Credits

- Model: **Qwen2.5-0.5B-Instruct** (Hugging Face).  
- Framework: **LLaMA-Factory** by *hiyouga*.  
- Dataset: **What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams**          
  (https://github.com/jind11/MedQA?tab=readme-ov-file)
  
---

**Enjoy building the Chinese Medical QA system with Fine-tuning & RAG!**
