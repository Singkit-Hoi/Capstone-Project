# 基于微调与 RAG 的中文医疗问答模型研究

> 在 Chinese MedQA 数据集上微调轻量级指令模型（Qwen2.5-0.5B-Instruct），并通过三种方式评测（直接调用、HTTP API、API + RAG），另外提供一个可通过 SSH 隧道在本地访问的 Gradio 界面。

---

## 1）概览（Overview）

本仓库提供了一条端到端（end-to-end）的工作流，用于：
- 将原始 MedQA（简体中文子集）转换为 **SFT** 格式；
- 使用 **LLaMA-Factory** 对 **Qwen2.5-0.5B-Instruct** 进行微调，支持 **LoRA** 与**全量微调（full-parameter training）**；
- 以三种方式评测模型效果：直接调用、HTTP API 调用（将训练/未训练模型部署为 HTTP API 服务）、以及 **API + RAG**；
- 运行交互式 **Gradio** 演示，并通过 **SSH Tunneling Tool** 从本地机器访问；
- （可选）将模型接入 **Dify**（YouTube 链接：https://youtu.be/qT1t545kN6I）。

---

## 2）目录结构（示例）

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

**教材下载**: LLaMA-Factory/2025_05_22_med_data_zh_paragraph 中的教材不完整，请通过 https://github.com/jind11/MedQA?tab=readme-ov-file google drive 下载 (data_clean/data_clean/zh_paragrah)

---

## 3）先决条件（Prerequisites）

- **GPU**：NVIDIA，显存充足（建议 ≥4 GB 以支持全量微调；LoRA 更轻量）；
- **OS / CUDA / Python**：AutoDL（或同类环境），Python 3.10+，并安装与 CUDA 兼容的 PyTorch；
- **Git LFS** 与（可选）**Hugging Face CLI** 用于模型下载。

---

## 4）外部下载（需预先准备）

1. **Qwen2.5-0.5B-Instruct** —— 从 **Hugging Face** 下载并放置到：
   ```text
   ./Qwen2.5-0.5B-Instruct
   ```

2. **LLaMA-Factory** —— 克隆与安装：  
   https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md
   ```bash
   git clone https://github.com/hiyouga/LLaMA-Factory.git autodl-tmp/LLaMA-Factory
   cd autodl-tmp/LLaMA-Factory
   pip install -e ".[torch,metrics,quality]"
   ```

> 下面未特别说明的命令，均在 `autodl-tmp/LLaMA-Factory` 目录中执行。

---

## 5）数据准备（转为 SFT）

1. 将 **MedQA（简体中文子集）** 的原始数据放至 `LLaMA-Factory/data/`；  
2. 运行转换脚本，生成 SFT 格式：
   ```bash
   cd autodl-tmp/LLaMA-Factory/data
   python process_data.py
   ```

脚本会生成符合 SFT 结构的 `train.jsonl`。

---

## 6）训练配置（Configure Training）

请根据实际环境与路径，编辑以下 YAML：
- **LoRA**：`examples/train_lora/llama3_lora_sft.yaml`
- **全量微调**：`examples/train_full/llama3_full_sft.yaml`

> 具体参数以 YAML 文件为准。

---

## 7）训练与导出（Training & Export）

在 `autodl-tmp/LLaMA-Factory` 根目录执行：

### A）LoRA 微调与导出
```bash
# 训练 LoRA
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

# 合并 / 导出 LoRA 权重
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

### B）全量微调
```bash
llamafactory-cli train examples/train_full/llama3_full_sft.yaml
# （检查点会保存到 YAML 配置的输出目录）
```

---

## 8）评测（三种方式）

评测代码位于 `evaluation/`：

### 8.1 直接评估（不启服务）
- 目录：`evaluation/Direct Evaluation`
- 示例：`evaluate_base_model.py`（在脚本中将 `model_path` 设置为基座或微调后的模型路径）
```bash
cd evaluation/Direct\ Evaluation
python evaluate_base_model.py
```

### 8.2 HTTP API 评估（vLLM 后端）
1）**从 LLaMA-Factory 根目录**将模型以 HTTP API 形式提供：
```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3.yaml   infer_backend=vllm vllm_enforce_eager=true
# 若为微调模型，请切换为对应 YAML：
#   examples/inference/llama3_lora_sft.yaml
#   examples/inference/llama3_full_sft.yaml
```

2）运行评测脚本：
- 目录：`evaluation/API Evaluation`
- 示例：`base_API_evaluation.py`（确保所服务的 YAML 与目标模型一致）
```bash
cd evaluation/"API Evaluation"
python base_API_evaluation.py
```

### 8.3 HTTP API + RAG 评估
- 目录：`evaluation/API with Rag Evaluation`
- 示例：`Rag_model_evaluation.py`（与 8.2 相同的模型服务，外加检索配置）
```bash
cd evaluation/"API with Rag Evaluation"
python Rag_model_evaluation.py
```

---

## 9）交互式演示（Gradio）与 SSH 隧道

1. **本地机器上的 SSH Tunneling Tool**  
   - 填写实例的 SSH 命令与密码；  
   - 将 **“Proxy to local port” 设为 `7860`**，保留远端端口（如 `1080`）。  
2. **AutoDL 实例上**（进入 `LLaMA-Factory/`）：
   ```bash
   python Gradio_demo_new.py
   ```
3. 回到隧道工具 → 点击 **Start Proxy**；  
4. 在浏览器打开 **http://127.0.0.1:7860**，即可在本地访问远端 Gradio UI。

---

## 10）Dify 集成（可选）

可将 Dify 指向正在运行的 HTTP API 端点以完成对接。  
**YouTube 链接**：*(https://youtu.be/qT1t545kN6I)*。

---

## 11）快速开始（可复制粘贴）

```bash
# 克隆 LLaMA-Factory 并安装
cd ~/autodl-tmp
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,quality]"

# 准备 Qwen2.5-0.5B-Instruct：~/autodl-tmp/Qwen2.5-0.5B-Instruct

# 将数据转为 SFT
cd data && python process_data.py 

# 编辑 examples/train_lora 与 examples/train_full 下的 YAML

# 训练与导出
llamafactory-cli train  examples/train_lora/llama3_lora_sft.yaml
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
llamafactory-cli train  examples/train_full/llama3_full_sft.yaml

# 启动 API（从如下 YAML 中选择其一）
API_PORT=8000 llamafactory-cli api examples/inference/llama3_full_sft.yaml infer_backend=vllm vllm_enforce_eager=true

# 评测
python evaluation/Direct\ Evaluation/evaluate_base_model.py
python evaluation/"API Evaluation"/base_API_evaluation.py
python evaluation/"API with Rag Evaluation"/Rag_model_evaluation.py

# 在服务器上运行 Gradio，然后通过 SSH 隧道访问 http://127.0.0.1:7860
```

---

## 12）故障排查（Troubleshooting）

- **CUDA OOM**：减小 `micro_batch_size`，启用 gradient checkpointing，或从 LoRA 方案入手；  
- **API 不可访问**：确认 `llamafactory-cli api ...` 进程已在端口 `8000` 运行；  
- **RAG 报错**：检查你的文档目录是否存在，以及 embedder / checkpoint 的路径是否正确；  
- **路径问题**：再次核对 YAML 中的 `model_name_or_path`、`data_path` 与 `output_dir`。

---

## 13）许可与致谢（License & Credits）

- 模型：**Qwen2.5-0.5B-Instruct**（Hugging Face）；  
- 框架：**LLaMA-Factory**（作者：*hiyouga*）。
- 数据集: **What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams**
  (https://github.com/jind11/MedQA?tab=readme-ov-file)
---

**祝你在 Fine-tuning 与 RAG 的中文医疗问答系统搭建之旅一路顺利！**
