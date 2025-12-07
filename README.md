# Efficient Fine-Tuning Strategies for BERT

### A Systematic Study on SST-2 with PEFT, Data-Centric Learning, and Transfer Learning

**Course Project for AIAA3102, HKUST(GZ)**

**Authors:** Boyi Zhang*, Weifeng Chen*, Yunxiang Li
(* Equal Contribution)

---

## 1. Overview

This project systematically studies **efficient fine-tuning strategies for Transformer-based sentiment classification under realistic GPU and engineering constraints**. Motivated by a startup-style deployment scenario with limited computational resources, we investigate:

* **Full fine-tuning**
* **LoRA fine-tuning (16-bit / 8-bit)**
* **QLoRA fine-tuning (4-bit)**

on the **SST-2 sentiment classification benchmark**, using `bert-base-uncased` as the backbone. Beyond efficiency benchmarking, we further conduct controlled studies on:

* Hyperparameter sensitivity (learning rate, LoRA rank)
* Robustness to randomness and noisy text
* Few-shot generalization and data scaling behavior
* Data-centric difficulty-based sampling
* Model capacity and overfitting via layer freezing
* Cross-task transfer learning from SST-2 to MRPC
* SVD-based stable-rank analysis of LoRA updates
* A deployable **Gradio-based interactive evaluation system**

This repository contains **all training, evaluation, data-centric, transfer-learning, and demo code**, together with **logged experimental results** and **visualization pipelines**.

---

## 2. Reproducibility Policy

We follow standard **reproducible ML experiment practices**:

* All experiments are **configuration-driven** via `src/config.py`
* All metrics are **logged to CSV or JSON**
* All figures are **generated from logged CSV or JSON**
* No manual reporting of numbers
* Each experiment is **deterministic given a random seed using `transformers.set_seed()`**
* Multi-seed results are evaluated using **bootstrap confidence intervals**

The codebase supports **full end-to-end reproduction** of all figures and tables reported in the project report.

---

## 3. Repository Structure

```text
PJ/
├── models/                     
│   ├── Full Finetune (FP16)/
│   ├── Full Finetune (FP32)/
│   ├── LoRA (FP16)/
│   ├── LoRA (8-bit)/
│   └── QLoRA (4-bit)/
│
├── notebooks/
│   ├── main.ipynb              
│   └── experiments.ipynb      
│
├── results/
│   ├── csvs/                   
│   ├── figures/               
│   └── transfer learning/     
│
├── src/
│   ├── app.py                  
│   ├── config.py               
│   ├── utils.py                
│   ├── data.py                
│   ├── noisy_data.py           
│   ├── data_centric.py         
│   └── transfer.py             
│
├── requirements.txt
└── README.md
```

---

## 4. Functional Responsibilities

| Module            | Function                                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `config.py`       | Centralized default configuration of model names, LoRA hyperparameters, dataset variants, random seeds, DeepSeek API key |
| `utils.py`        | Unified pipeline for dataset loading, tokenizer construction, model initialization, training, evaluation, and logging         |
| `noisy_data.py`   | Class-based implementation of character-level and word-level corruption                                                       |
| `data_centric.py` | Difficulty-based data selection and sampling strategies                                                                       |
| `transfer.py`     | Cross-task transfer learning and SVD-based LoRA update analysis                                                               |
| `app.py`          | Gradio-based inference and evaluation interface                                                                               |

---

## 5. Entry-Point Design

### 5.1 `notebooks/main.ipynb` (Instructor / TA Execution Entry)

Designed under the **2-hour Colab runtime constraint**, this notebook provides:

* ✅ Figure generation from pre-logged CSVs
* ✅ Training of **one single model instance (~30 min)**
* ✅ Launching the **Gradio interactive system**

This is the **official execution entry** for evaluation and demonstration.

---

### 5.2 `notebooks/experiments.ipynb` (Full Experimental Suite)

Contains all heavyweight experiments:

* Efficiency benchmarking (Full vs LoRA vs QLoRA)
* Hyperparameter sweeps
* Robustness analysis
* Few-shot scaling
* Data-centric learning
* Capacity control
* Transfer learning
* SVD-based LoRA analysis

This notebook is **not required for real-time demonstration**, but supports **full methodological replication**. Please refer to it for comprehensive experimental details Please notice that you must have at least 300GB of free Google Drive space if you want to run it on Google Colab with mounting to Google Drive.

---

## 6. Supported Fine-Tuning Strategies

| Method         | Backbone Precision | Quantization | Trainable Ratio |
| -------------- | ------------------ | ------------ | --------------- |
| Full Fine-Tune | FP32 / FP16        | None         | ~100%           |
| LoRA (16-bit)  | FP16 / BF16        | None         | ~0.8%           |
| LoRA (8-bit)   | INT8               | 8-bit        | ~0.8%           |
| QLoRA (4-bit)  | INT4               | 4-bit        | ~1.3%           |

All methods use the **same classifier head and tokenization pipeline** to ensure controlled comparison.

---

## 7. Data-Centric Learning Protocol

We construct a **teacher model** using full-data training and compute:

* Per-sample cross-entropy loss
* Difficulty ranking within each label
* Difficulty-stratified subset sampling

We compare:

* Uniform random sampling
* Only easy samples
* Only hard samples
* Middle-only sampling
* **Mixed-difficulty sampling (proposed)**

---

## 8. Transfer Learning Protocol

* **Source Task:** SST-2

* **Target Task:** MRPC

* **Method:**

  1. Train LoRA on SST-2
  2. Transfer LoRA weights to MRPC
  3. Freeze backbone and adapters
  4. Train classifier head only

* Baseline: LoRA trained from scratch on MRPC

* Adapter placements: `attn-qkv`, `attn-full`, `ffn-only`, `full-model`

**LoRA update matrices ΔW = BA are further analyzed using:**

* Per-layer SVD
* Stable-rank statistics
* Cross-task transfer correlation

Implementation:

```
src/transfer.py
```

---

## 9. Gradio Interactive Evaluation System

Implemented in:

```
src/app.py
```

### Supported Features

* Model selection (Full / LoRA / QLoRA / API)
* Single-text prediction
* Softmax confidence reporting
* Real-time latency measurement
* Batch CSV evaluation
* Accuracy & mean latency summary
* Model parameter display

This interface is designed for **non-technical stakeholders** to directly evaluate accuracy–latency–cost trade-offs.

---

## 10. Installation

```bash
pip install -r requirements.txt
```

### Dependency List

```text
torch
peft
numpy
scikit-learn
transformers
datasets
accelerate
evaluate
bitsandbytes
gradio
seaborn
pandas
matplotlib
tqdm
tabulate
GPUtil
openai
scipy
```

---

## 11. Execution Protocol (Standardized)

### 11.1 Single-Model Training (Demonstration)

Please refer to `notebooks/main.ipynb` *Section 2* for detailed instructions.

### 11.2 Launch Gradio Interface

```bash
#activate your conda env if needed
cd src
python app.py
```

Access via:

```
http://127.0.0.1:7860(just a reference)
```
if running locally.

Alternatively, use the public share link if running on Colab.

---

## 12. Threats to Validity

* All main experiments are conducted on **SST-2**, a clean and short-text benchmark.
* Only **one cross-task transfer target (MRPC)** is explored.
* Noise modeling is **synthetic character/word-level corruption**.
* LoRA update SVD analysis provides **correlation, not causality**.
* Large-scale LLMs (≥7B) are not tested due to compute limits.

---

## 13. Practical Recommendations

For real-world sentiment classification under limited resources:

* ✅ Use **BERT + LoRA (16-bit)**
* ✅ LoRA rank = 16
* ✅ Learning rate ≈ 1e-4
* ✅ Early stopping within 5–10 epochs
* ✅ ~1000 clean labeled examples are often sufficient
* ✅ Use distribution-preserving data selection
* ✅ Design adapters covering **both attention and FFN**

---

## 14. Team Contribution

| Member           | Contribution                                                                                                                            |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Weifeng Chen** | Project lead; PEFT benchmarking; robustness analysis; capacity control; transfer learning; SVD analysis; report writing; demo recording |
| **Boyi Zhang**   | Data-centric learning; scaling experiments; visualization; presentation slides; report writing                                          |
| **Yunxiang Li**  | Gradio frontend implementation; full experiment reproduction; README documentation                                                      |

