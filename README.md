# 🧠 Papers_QA: Medical Paper Question Answering System

**Papers_QA** is an end-to-end pipeline for extracting question-answer pairs from medical research papers. It leverages large language models and dense retrieval to automate medical information understanding and evaluation.

---

## 🚀 Features

- 🧾 **Mistral-7B-Instruct (4-bit)** for answer generation
- 🔍 **Sentence-BERT + FAISS** for dense passage retrieval
- 🧪 **BLEU scoring and retrieval accuracy** for evaluation
- 📊 Automatically generated training data
- 🧠 Supports fine-tuning, inference, and evaluation

---

## 📁 Project Structure

```plaintext
📦 Papers_QA
├── data/
│   └── generated/
│       └── train_data.csv          # Automatically generated QA pairs
├── notebooks/
│   ├── 1_qa_generation.ipynb       # Generate QAs from medical papers
│   ├── 3_inference.ipynb           # Retrieval + inference + BLEU evaluation
│   └── medqa_training.ipynb        # Fine-tuning Mistral (optional)
├── docs/
│   └── MedQA_Documentation.pdf     # Project overview/report
├── src/                            # Placeholder for reusable code modules
├── requirements.txt                # Project dependencies
├── LICENSE                         # MIT License
└── README.md                       # This file
