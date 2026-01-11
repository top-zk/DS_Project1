---
language: zh
license: apache-2.0
---

# Bert-base-chinese

This is the `bert-base-chinese` model, which serves as the **backbone encoder** for the **Intelligent Medical Diagnostic Assistant**.

## 🚀 Application in Medical Diagnosis

In this project, this pre-trained model is fine-tuned to classify medical symptoms into 10 distinct departments/categories.

### Key Modifications
*   **Fine-tuning Task**: Sequence Classification (10 classes).
*   **Input**: Natural language description of symptoms (Chinese).
*   **Output**: Probability distribution over disease categories (e.g., Respiratory, Cardiology, etc.).

### Integration
This model works in tandem with a **Rule-based Keyword Matching System** to ensure high accuracy for common, specific symptoms while leveraging BERT's semantic understanding for complex descriptions.

## Table of Contents
- [Model Details](#model-details)
- [Uses](#uses)
- [Risks, Limitations and Biases](#risks-limitations-and-biases)
- [Training](#training)
- [Evaluation](#evaluation)
- [How to Get Started With the Model](#how-to-get-started-with-the-model)


## Model Details

### Model Description

This model has been pre-trained for Chinese, training and random input masking has been applied independently to word pieces (as in the original BERT paper).

- **Developed by:** Google
- **Model Type:** Fill-Mask (Pre-training), Sequence Classification (Fine-tuning)
- **Language(s):** Chinese
- **License:** Apache 2.0
- **Parent Model:** See the [BERT base uncased model](https://huggingface.co/bert-base-uncased) for more information about the BERT base model.

### Model Sources
- **GitHub repo**: https://github.com/google-research/bert/blob/master/multilingual.md
- **Paper:** [BERT](https://arxiv.org/abs/1810.04805)


## Uses

#### Direct Use

This model can be used for masked language modeling 

#### Downstream Use (Current Project)

Used for **Symptom-to-Department Classification**:
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the fine-tuned model (or base model for inference if weights are loaded)
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=10)

text = "我最近总是头痛，晚上睡不着觉"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
# Returns probability of [Respiratory, Cardiology, Gastroenterology, Neurology, ...]
```


## Risks, Limitations and Biases
**CONTENT WARNING: Readers should be aware this section contains content that is disturbing, offensive, and can propagate historical and current stereotypes.**

Significant research has explored bias and fairness issues with language models (see, e.g., [Sheng et al. (2021)](https://aclanthology.org/2021.acl-long.330.pdf) and [Bender et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922)).


## Training

#### Training Procedure
* **type_vocab_size:** 2
* **vocab_size:** 21128
* **num_hidden_layers:** 12

#### Training Data
[More Information Needed]

## Evaluation

#### Results

[More Information Needed]


## How to Get Started With the Model
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

```