# MickTransformer 🚀

**Building and Training a Small Transformer Language Model**

A machine learning research project using Python and PyTorch.

## About This Project

MickTransformer is a small character-level Transformer language model.

The goal of this project is to understand how Transformers work, train a model from scratch using PyTorch components, investigate problems during development, and explore training dynamics through controlled experiments.

## Task A — Build and Train a Transformer

### 1. Model Architecture

My Transformer includes:

- Character-level Tokenization
- Token Embedding
- Positional Embedding
- Single-Head Causal Self-Attention
- Query, Key, and Value
- MLP (Multilayer Perceptron)
- Residual Connections and Layer Normalization
- Logits and Cross-Entropy Loss
- Backpropagation
- Adam Optimizer

**Training pipeline:**

Token → Embedding → Attention → MLP → Logits → Loss → Backward → Optimizer

### 2. Training Results

The model was trained on:

`the cat sleeps.`

After 200 optimization steps:

| Step | Training Loss |
|---|---|
| 0 | 2.3910 |
| 20 | 0.7567 |
| 40 | 0.1332 |
| 80 | 0.0177 |
| 120 | 0.0088 |
| 160 | 0.0056 |
| 200 | 0.0039 |

**Result:** Training Loss decreased by approximately 99.84%.

### 3. Text Generation

Input: `the `

Generated output: `the cat sleeps.`

The Transformer successfully reproduced the training sentence.

However, this does not prove the model can generalize to unseen text.

### 4. Inference Optimization Experiment

I compared regular forward computation with PyTorch's inference mode.

| Measurement | Result |
|---|---|
| Normal execution | 0.01727 seconds |
| Optimized execution | 0.01529 seconds |
| Speedup | 1.13× |
| Predictions match | True |

Inference mode was faster in this small benchmark. Additional measurements would be needed to confirm how consistent the improvement is.

### 5. Debugging

During development, I encountered an indentation problem in my Python training loop.

I corrected the indentation of the optimizer operations and verified the updated program by running it again.

See `debug-note.md` for the debugging record.

## Task B — Training Dynamics Research

**Research Question:**

Does increasing context length help a Transformer predict the next character more accurately?

### Experimental Plan

Compare two context lengths:

- Experiment A: 4 tokens
- Experiment B: 12 tokens

Keep other training conditions consistent and compare validation loss.

### Research Process

1. Write a hypothesis before testing.
2. Explain the reasoning behind the hypothesis.
3. Design controlled experiments.
4. Collect training and validation measurements.
5. Investigate unexpected results.
6. Explain conclusions and remaining uncertainties.

**Status: In Progress**

The experiment results and conclusions will be added after the experiments have been completed.

## How to Run

### Requirements

- Python 3
- PyTorch

Install PyTorch:

`python -m pip install torch`

Run the Transformer:

`python train.py`

The script runs training, text generation, and inference benchmarking.

## Project Files

| File | Description |
|---|---|
| `train.py` | Main Transformer implementation |
| `debug-note.md` | Task A debugging report |
| `task_a_run.log` | Training and optimization results |
| `task_b_report.md` | Task B research report (in progress) |
| `Add.py` | Additional development script |

## Limitations

This is a very small experimental model trained on a short sentence.

A low training loss does not necessarily mean the model understands language or generalizes well.

More diverse training data, held-out validation data, and repeated experiments are needed for stronger conclusions.

## Research Reflection

**What I learned:**  
[Write this in my own words.]

**What went wrong:**  
[Describe an actual difficulty and how I investigated it.]

**What I still don't know:**  
[Explain a question I genuinely haven't answered yet.]

## Author

**Mick Zhang**

Independent Developer | Game Development | Artificial Intelligence

GitHub: [Mick666-pro](https://github.com/Mick666-pro)

---

*This project is part of my hands-on learning and research into Transformers, machine learning, and training dynamics.*